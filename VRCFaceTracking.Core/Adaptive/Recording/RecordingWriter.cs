using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using VRCFaceTracking.Core.Params.Data.Mutation;

namespace VRCFaceTracking.Core.Adaptive.Recording;

/// <summary>
/// Writes <see cref="RecordingFormat"/> files for a <see cref="SessionRecorder"/>. Producers (tracking thread, mic
/// thread, UI) only enqueue and never block: a full queue drops the record and counts it. One background thread
/// compresses and writes, opens a file on the first frame, rotates it hourly, closes it when the recorder is switched
/// off or frames stop for a minute, and logs tracking/focus/drop events and mutations switched on or off.
/// </summary>
public sealed class RecordingWriter
{
    public static TimeSpan RotateAfter = TimeSpan.FromHours(1);
    public static TimeSpan IdleCloseAfter = TimeSpan.FromSeconds(60);
    public static long MinFreeBytes = 2L * 1024 * 1024 * 1024;   // Never fill the recordings drive past 2 GB free.
    private static readonly TimeSpan SpaceCheckEvery = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan FlushEvery = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan FocusProbeEvery = TimeSpan.FromSeconds(1);
    private static readonly TimeSpan ErrorCooldown = TimeSpan.FromSeconds(30);
    private static readonly TimeSpan FrozenAfter = TimeSpan.FromSeconds(0.5);   // DESIGN.md 6.1
    private const float FrozenEpsilon = 1e-4f;
    private const int QueueCapacity = 8192;   // ~40 s of 90 Hz frames plus 100 Hz mic levels

    private readonly record struct Record(byte Type, long Tick, float[]? Frame, float Level, string? Text);

    private readonly SessionRecorder _owner;
    private readonly ILogger _logger;
    private readonly BlockingCollection<Record> _queue = new(new ConcurrentQueue<Record>(), QueueCapacity);
    private readonly Thread _thread;
    private readonly byte[] _scratch = new byte[16];
    private long _dropped;
    private int _shutdown;

    // Session = one app run; it can span several files (rotation, idle gaps).
    private readonly string _sessionId = Guid.NewGuid().ToString("N");
    private long _sessionStartTick;
    private int _fileIndex;
    private string? _previousFileName;

    // Current file.
    private Stream? _out;
    private string? _path;
    private string? _directory;
    private volatile bool _recording;
    private long _fileStartTick, _lastFrameTick, _lastFlushTick, _cooldownUntil;
    private long _frames, _levels, _markers, _droppedReported;

    // Gate and side-probe state.
    private readonly float[] _shapeSnapshot = new float[RecordingFormat.ShapeCount];
    private long _lastShapeChangeTick;
    private string _trackingState = "ok";
    private string? _focus;
    private long _nextFocusProbe;
    private string? _activeMutations;
    private MicLevelMonitor? _mic;
    private string _micFilter = "";
    private string? _spaceCheckedDirectory;
    private long _nextSpaceCheck;
    private bool _lowDisk;

    public RecordingWriter(SessionRecorder owner, ILogger logger)
    {
        _owner = owner;
        _logger = logger;
        _thread = new Thread(Run) { IsBackground = true, Name = "VRCFT Session Recorder" };
        _thread.Start();
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Shutdown(TimeSpan.FromSeconds(2));
    }

    public bool IsRecording => _recording;

    public long DroppedRecords => Interlocked.Read(ref _dropped);

    public void EnqueueFrame(long tick, float[] frame)
    {
        if (!TryEnqueue(new Record(RecordingFormat.FrameRecord, tick, frame, 0, null)))
        {
            ArrayPool<float>.Shared.Return(frame);
        }
    }

    public void EnqueueLevel(long tick, float dbfs) =>
        TryEnqueue(new Record(RecordingFormat.AudioRecord, tick, null, dbfs, null));

    public void EnqueueMarker(string label) =>
        TryEnqueue(new Record(RecordingFormat.MarkerRecord, Stopwatch.GetTimestamp(), null, 0, label));

    public void EnqueueEvent(string text) =>
        TryEnqueue(new Record(RecordingFormat.GateRecord, Stopwatch.GetTimestamp(), null, 0, text));

    /// <summary> Writes what is queued, closes the file and stops the thread. Safe to call more than once. </summary>
    public void Shutdown(TimeSpan timeout)
    {
        if (Interlocked.Exchange(ref _shutdown, 1) == 0)
        {
            _queue.CompleteAdding();
        }
        _thread.Join(timeout);
    }

    private bool TryEnqueue(Record record)
    {
        try
        {
            if (_queue.TryAdd(record))
            {
                return true;
            }
        }
        catch (InvalidOperationException)
        {
            return false;   // Shutting down: not a drop.
        }
        Interlocked.Increment(ref _dropped);
        return false;
    }

    private void Run()
    {
        long nextHousekeeping = 0;
        while (!_queue.IsCompleted)
        {
            if (_queue.TryTake(out var record, 100))
            {
                try
                {
                    Handle(record);
                }
                catch (Exception e)
                {
                    Fail(e);
                }
                finally
                {
                    if (record.Frame != null)
                    {
                        ArrayPool<float>.Shared.Return(record.Frame);
                    }
                }
            }

            var now = Stopwatch.GetTimestamp();
            if (now < nextHousekeeping)
            {
                continue;
            }
            nextHousekeeping = now + Stopwatch.Frequency / 10;

            try
            {
                Housekeeping(now);
            }
            catch (Exception e)
            {
                Fail(e);
            }
        }

        try
        {
            StopMic();
            CloseFile("exit");
        }
        catch (Exception e)
        {
            _logger.LogError(e, "Session Recorder could not close its file cleanly.");
        }
    }

    private void Handle(Record record)
    {
        if (record.Type == RecordingFormat.FrameRecord)
        {
            if (!_owner.IsActive || record.Tick < _cooldownUntil)
            {
                return;
            }

            if (_out == null)
            {
                if (!HasRoom(_owner.RecordingsDirectory, record.Tick))
                {
                    return;
                }
                OpenFile(record.Tick);
            }
            else if (record.Tick - _fileStartTick >= Ticks(RotateAfter))
            {
                CloseFile("rotate");
                OpenFile(record.Tick);
            }

            TrackTrackingState(record.Tick, record.Frame!);
            WriteTick(RecordingFormat.FrameRecord, record.Tick);
            _out!.Write(MemoryMarshal.AsBytes(record.Frame.AsSpan(0, RecordingFormat.FrameFloats)));
            _lastFrameTick = record.Tick;
            _frames++;
            return;
        }

        if (_out == null)
        {
            return;   // Not recording: mic levels and events have nowhere to go.
        }

        switch (record.Type)
        {
            case RecordingFormat.AudioRecord:
                WriteTick(RecordingFormat.AudioRecord, record.Tick);
                BinaryPrimitives.WriteSingleLittleEndian(_scratch, record.Level);
                _out.Write(_scratch, 0, 4);
                _levels++;
                break;
            case RecordingFormat.MarkerRecord:
                WriteText(RecordingFormat.MarkerRecord, record.Tick, record.Text!);
                _markers++;
                break;
            case RecordingFormat.GateRecord:
                WriteText(RecordingFormat.GateRecord, record.Tick, record.Text!);
                break;
        }
    }

    private void Housekeeping(long now)
    {
        if (_out != null && !_owner.IsActive)
        {
            CloseFile("disabled");
        }
        else if (_out != null && !HasRoom(_directory!, now))
        {
            CloseFile("low_disk");
        }
        else if (_out != null && now - _lastFrameTick > Ticks(IdleCloseAfter))
        {
            CloseFile("idle");
        }

        ReconcileMic();

        if (_out == null)
        {
            return;
        }

        var dropped = Interlocked.Read(ref _dropped);
        if (dropped != _droppedReported)
        {
            WriteText(RecordingFormat.GateRecord, now, $"recorder:dropped {dropped - _droppedReported}");
            _droppedReported = dropped;
        }

        // A mutation switched on or off since the header (Adaptive Calibration, the stock Calibration in an A/B, ...).
        if (DescribeActiveMutations() is { } active && active != _activeMutations)
        {
            _activeMutations = active;
            WriteText(RecordingFormat.GateRecord, now, "mutations:" + active);
        }

        if (now >= _nextFocusProbe)
        {
            _nextFocusProbe = now + Ticks(FocusProbeEvery);
            var focus = ForegroundApp.Classify();
            if (focus != _focus)
            {
                _focus = focus;
                WriteText(RecordingFormat.GateRecord, now, "focus:" + focus);
            }
        }

        if (now - _lastFlushTick >= Ticks(FlushEvery))
        {
            Flush(now);
        }
    }

    // DESIGN.md 6.1: all-zero frames (XR_FB_face_tracking2 zeroes weights on tracking loss) and frozen data (no
    // expression moved more than 1e-4 for 0.5 s). Logged as transitions; offline code can recompute from the frames.
    private void TrackTrackingState(long tick, float[] frame)
    {
        var allZero = true;
        var changed = false;
        var raw = frame.AsSpan(RecordingFormat.Channels.Length, RecordingFormat.ShapeCount);
        for (var i = 0; i < raw.Length; i++)
        {
            var value = raw[i];
            allZero &= value == 0f;
            changed |= Math.Abs(value - _shapeSnapshot[i]) > FrozenEpsilon;
        }

        if (changed)
        {
            raw.CopyTo(_shapeSnapshot);
            _lastShapeChangeTick = tick;
        }

        var state = allZero ? "zeros" : tick - _lastShapeChangeTick >= Ticks(FrozenAfter) ? "frozen" : "ok";
        if (state != _trackingState)
        {
            _trackingState = state;
            WriteText(RecordingFormat.GateRecord, tick, "tracking:" + state);
        }
    }

    private void OpenFile(long firstTick)
    {
        var directory = _owner.RecordingsDirectory;
        Directory.CreateDirectory(directory);

        var stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        var path = Path.Combine(directory, stamp + RecordingFormat.FileExtension);
        for (var n = 1; File.Exists(path); n++)
        {
            path = Path.Combine(directory, $"{stamp}-{n}{RecordingFormat.FileExtension}");
        }

        if (_sessionStartTick == 0)
        {
            _sessionStartTick = firstTick;
        }

        _activeMutations = DescribeActiveMutations(); // before the header: a change in between is logged, never lost
        var header = BuildHeader(firstTick);
        var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 1 << 16);
        try
        {
            file.Write(Encoding.UTF8.GetBytes(header.ToString(Formatting.None) + "\n"));
            _out = new BufferedStream(new DeflateStream(file, CompressionLevel.Optimal, leaveOpen: false), 1 << 16);
        }
        catch
        {
            file.Dispose();
            throw;
        }

        _path = path;
        _directory = directory;
        _fileIndex++;
        _fileStartTick = _lastFrameTick = _lastFlushTick = firstTick;
        _frames = _levels = _markers = 0;
        _droppedReported = Interlocked.Read(ref _dropped);
        Array.Clear(_shapeSnapshot);
        _lastShapeChangeTick = firstTick;
        _trackingState = "ok";
        _focus = null;
        _nextFocusProbe = 0;
        if (_mic != null && OperatingSystem.IsWindows())
        {
            // Rotation keeps the mic running; note its device at the top of the new file.
            WriteText(RecordingFormat.GateRecord, firstTick, "audio:on " + _mic.DeviceDescription);
        }
        _recording = true;
        _logger.LogInformation("Session Recorder: recording to {Path}", path);
    }

    private void CloseFile(string reason)
    {
        if (_out == null)
        {
            return;
        }

        var seconds = (_lastFrameTick - _fileStartTick) / (double)Stopwatch.Frequency;
        try
        {
            WriteText(RecordingFormat.GateRecord, Stopwatch.GetTimestamp(), "recorder:stop " + reason);
        }
        finally
        {
            try
            {
                _out.Dispose();   // Ends the deflate stream and closes the file.
            }
            finally
            {
                _out = null;
                _recording = false;
                _previousFileName = Path.GetFileName(_path);
            }
        }

        _logger.LogInformation(
            "Session Recorder: closed {Path} ({Reason}): {Seconds:F0} s, {Frames} frames, {Levels} mic levels, {Markers} markers, {Dropped} dropped so far",
            _path, reason, seconds, _frames, _levels, _markers, Interlocked.Read(ref _dropped));
    }

    private void Flush(long now)
    {
        _out!.Flush();   // BufferedStream -> DeflateStream sync flush -> FileStream
        _lastFlushTick = now;
    }

    // Keeps the recordings drive from filling up: below MinFreeBytes no file opens and an open one closes; recording
    // resumes by itself once there is room again. Checked at most every 30 s per folder.
    private bool HasRoom(string directory, long now)
    {
        if (now < _nextSpaceCheck && directory == _spaceCheckedDirectory)
        {
            return !_lowDisk;
        }
        _nextSpaceCheck = now + Ticks(SpaceCheckEvery);
        _spaceCheckedDirectory = directory;

        string drive;
        long free;
        try
        {
            drive = Path.GetPathRoot(Path.GetFullPath(directory))!;
            free = new DriveInfo(drive).AvailableFreeSpace;
        }
        catch (Exception)
        {
            return true;   // Network path or missing drive: OpenFile reports the real error.
        }

        var low = free < MinFreeBytes;
        if (low && !_lowDisk)
        {
            _logger.LogWarning(
                "Session Recorder paused: only {FreeGb:F1} GB free on {Drive} (it keeps at least {MinGb:F0} GB free). It resumes when there is room.",
                free / 1e9, drive, MinFreeBytes / 1e9);
        }
        else if (!low && _lowDisk)
        {
            _logger.LogInformation("Session Recorder: enough free space on {Drive} again, resuming.", drive);
        }
        _lowDisk = low;
        return !low;
    }

    // Mic capture runs only while a file is open and the setting is on; a new device filter restarts it.
    private void ReconcileMic()
    {
        var want = _out != null && _owner.recordMicLevel && OperatingSystem.IsWindows();
        var filter = _owner.micDeviceFilter?.Trim() ?? "";
        if (_mic != null && (!want || filter != _micFilter))
        {
            StopMic();
            if (_out != null)
            {
                WriteText(RecordingFormat.GateRecord, Stopwatch.GetTimestamp(), "audio:off");
            }
        }

        if (_mic == null && want && OperatingSystem.IsWindows())
        {
            _micFilter = filter;
            _mic = new MicLevelMonitor(filter, EnqueueLevel, EnqueueEvent, _logger);
        }
    }

    private void StopMic()
    {
        if (_mic != null && OperatingSystem.IsWindows())
        {
            _mic.Dispose();
        }
        _mic = null;
    }

    private void Fail(Exception e)
    {
        _logger.LogError(e, "Session Recorder write failed; pausing recording for {Seconds} s.", ErrorCooldown.TotalSeconds);
        _cooldownUntil = Stopwatch.GetTimestamp() + Ticks(ErrorCooldown);
        try
        {
            StopMic();
            CloseFile("error");
        }
        catch
        {
            _out = null;
            _recording = false;
        }
    }

    private void WriteTick(byte type, long tick)
    {
        _scratch[0] = type;
        BinaryPrimitives.WriteInt64LittleEndian(_scratch.AsSpan(1), tick);
        _out!.Write(_scratch, 0, 9);
    }

    private void WriteText(byte type, long tick, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var length = Math.Min(bytes.Length, ushort.MaxValue);
        WriteTick(type, tick);
        BinaryPrimitives.WriteUInt16LittleEndian(_scratch, (ushort)length);
        _out!.Write(_scratch, 0, 2);
        _out.Write(bytes, 0, length);
    }

    private static long Ticks(TimeSpan span) => (long)(span.TotalSeconds * Stopwatch.Frequency);

    private JObject BuildHeader(long firstTick)
    {
        var entry = Assembly.GetEntryAssembly();
        return new JObject
        {
            ["format"] = "vftrec",
            ["schema"] = RecordingFormat.SchemaVersion,
            ["app"] = new JObject
            {
                ["name"] = Utils.DataFolderName,
                ["version"] = entry?.GetName().Version?.ToString(),
                ["informational_version"] = entry?.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion,
                ["core_version"] = typeof(RecordingFormat).Assembly.GetName().Version?.ToString(),
            },
            ["session"] = new JObject
            {
                ["id"] = _sessionId,
                ["qpc_start"] = _sessionStartTick,
                ["file_index"] = _fileIndex,
                ["previous_file"] = _previousFileName,
            },
            ["qpc_frequency"] = Stopwatch.Frequency,
            ["qpc_start"] = firstTick,
            // One QPC/wall-clock pair taken together, to place ticks in local time.
            ["clock"] = new JObject
            {
                ["qpc"] = Stopwatch.GetTimestamp(),
                ["utc"] = DateTime.UtcNow.ToString("O"),
                ["utc_offset_minutes"] = TimeZoneInfo.Local.GetUtcOffset(DateTime.UtcNow).TotalMinutes,
            },
            ["host"] = new JObject
            {
                ["os"] = RuntimeInformation.OSDescription,
                ["runtime"] = RuntimeInformation.FrameworkDescription,
            },
            ["modules"] = JArray.Parse(DescribeModules()),
            ["expressions"] = new JArray(RecordingFormat.ExpressionNames()),
            ["frame"] = new JObject
            {
                ["channels"] = new JArray(RecordingFormat.Channels),
                ["blocks"] = new JArray("raw", "out"),
                ["block_length"] = RecordingFormat.BlockLength,
                ["dtype"] = "float32",
            },
            ["records"] = new JObject
            {
                ["F"] = "int64 tick, float32[2 * block_length]: raw block then out block",
                ["A"] = "int64 tick (end of block), float32 RMS dBFS of a 10 ms mic block",
                ["M"] = "int64 tick, uint16 n, utf8[n] marker label",
                ["G"] = "int64 tick, uint16 n, utf8[n] event 'kind:detail'",
            },
            ["audio"] = new JObject
            {
                ["enabled"] = _owner.recordMicLevel,
                ["block_ms"] = RecordingFormat.AudioBlockMilliseconds,
                ["device_filter"] = _owner.micDeviceFilter ?? "",
            },
            ["mutations"] = DescribeMutations(),
        };
    }

    /// <summary> Loaded modules as a JSON array string (also used for "modules:" events). </summary>
    public static string DescribeModules()
    {
        var result = new JArray();
        try
        {
            var modules = UnifiedTracking.Mutator?.LibManager?.LoadedModulesMetadata.ToArray() ?? [];
            foreach (var module in modules)
            {
                result.Add(new JObject
                {
                    ["name"] = module.Name,
                    ["active"] = module.Active,
                    ["eye"] = module.UsingEye,
                    ["expression"] = module.UsingExpression,
                });
            }
        }
        catch (Exception)
        {
            // Collection changed under us (a "modules:" event follows) or no mutator yet.
        }
        return result.ToString(Formatting.None);
    }

    /// <summary> Active mutations in run order as a JSON array string ("mutations:" events); null if the list changed under us. </summary>
    private static string? DescribeActiveMutations()
    {
        try
        {
            var mutations = UnifiedTracking.Mutator?._mutations.ToArray() ?? [];
            return new JArray(mutations.Where(m => m.IsActive).Select(m => m.Name).ToArray()).ToString(Formatting.None);
        }
        catch (Exception)
        {
            return null;
        }
    }

    // Settings snapshot of every mutation in run order. "properties" holds all UI-exposed fields, statics included
    // (Newtonsoft skips static fields, e.g. the Data Filter's cutoffs); "state" is the mutation as saved, which for
    // the stock Calibration includes its learned maxima.
    private static JArray DescribeMutations()
    {
        var result = new JArray();
        TrackingMutation[] mutations;
        try
        {
            mutations = UnifiedTracking.Mutator?._mutations.ToArray() ?? [];
        }
        catch (Exception)
        {
            mutations = [];
        }

        foreach (var mutation in mutations)
        {
            var entry = new JObject
            {
                ["name"] = mutation.Name,
                ["type"] = mutation.GetType().FullName,
                ["step"] = mutation.Step.ToString(),
                ["order"] = mutation.Order,
                ["active"] = mutation.IsActive,
            };

            var properties = new JObject();
            const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
            foreach (var field in mutation.GetType().GetFields(all))
            {
                if (field.GetCustomAttribute<MutationPropertyAttribute>() == null)
                {
                    continue;
                }
                try
                {
                    var value = field.GetValue(field.IsStatic ? null : mutation);
                    properties[field.Name] = value == null ? JValue.CreateNull() : JToken.FromObject(value);
                }
                catch (Exception e)
                {
                    properties[field.Name] = "error: " + e.Message;
                }
            }
            entry["properties"] = properties;

            try
            {
                entry["state"] = JToken.Parse(JsonConvert.SerializeObject(mutation));
            }
            catch (Exception e)
            {
                entry["state"] = "error: " + e.Message;
            }

            result.Add(entry);
        }
        return result;
    }

    /// <summary> Which app has the foreground: "vrchat", "vrcft", "other" or "none". Only the category is kept. </summary>
    private static class ForegroundApp
    {
        private static uint _lastPid;
        private static string _lastCategory = "none";

        public static string Classify()
        {
            if (!OperatingSystem.IsWindows())
            {
                return "unknown";
            }

            var window = GetForegroundWindow();
            if (window == IntPtr.Zero)
            {
                return "none";
            }

            GetWindowThreadProcessId(window, out var pid);
            if (pid == _lastPid)
            {
                return _lastCategory;
            }

            string category;
            if (pid == Environment.ProcessId)
            {
                category = "vrcft";
            }
            else
            {
                try
                {
                    using var process = Process.GetProcessById((int)pid);
                    category = string.Equals(process.ProcessName, "VRChat", StringComparison.OrdinalIgnoreCase) ? "vrchat" : "other";
                }
                catch (Exception)
                {
                    category = "other";
                }
            }

            _lastPid = pid;
            _lastCategory = category;
            return category;
        }

        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    }
}
