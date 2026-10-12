using System.Buffers;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Newtonsoft.Json;
using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Data.Mutation;

namespace VRCFaceTracking.Core.Adaptive.Recording;

/// <summary>
/// Adaptive-calibration Phase 1 recorder. Taps every tracking frame twice: as the module delivered it (this mutation
/// runs first) and as it leaves the mutator (<see cref="IMutationOutputObserver"/>), plus mic loudness, markers and
/// gate events, for offline tuning with tools/replay. Files stay on this PC; see <see cref="RecordingFormat"/>.
/// </summary>
/// <remarks>
/// The tracking thread only copies floats into a pooled buffer and enqueues it; <see cref="RecordingWriter"/> does
/// everything else on its own thread. A recording file opens on the first frame after the recorder is switched on.
/// </remarks>
public class SessionRecorder : TrackingMutation, IMutationOutputObserver
{
    // A changed folder applies from the next file.
    [MutationProperty(@"Recordings folder (blank = Documents\VRCFaceTracking-Adaptive\recordings)")]
    public string recordingsFolder = "";

    [MutationProperty("Record mic loudness (level only, never audio)", true)]
    public bool recordMicLevel = true;

    [MutationProperty("Mic name contains (blank = Windows default communications mic)")]
    public string micDeviceFilter = "";

    [MutationProperty("Marker label")]
    public string markerLabel = "";

    public override string Name => "Session Recorder";

    public override string Description =>
        "Records raw and final tracking data plus mic loudness (no audio) for calibration tuning. Stays on this PC.";

    public override MutationPriority Step => MutationPriority.Preprocessor;

    // First of all mutations: the raw block must be exactly what the module sent.
    public override int Order => int.MinValue;

    public override bool IsSaved => true;

    /// <summary> The live recorder, for the main window's marker hotkey. </summary>
    public static SessionRecorder? Current { get; private set; }

    public static readonly string DefaultRecordingsDirectory = Path.Combine(Utils.UserAccessibleDataDirectory, "recordings");

    /// <summary> Where the next recording file goes. </summary>
    [JsonIgnore]
    public string RecordingsDirectory => string.IsNullOrWhiteSpace(recordingsFolder)
        ? DefaultRecordingsDirectory
        : Environment.ExpandEnvironmentVariables(recordingsFolder.Trim());

    private RecordingWriter? _writer;
    private float[]? _pending;
    private long _pendingTick;
    private int _markerCount;
    private bool _faulted;

    public SessionRecorder()
    {
        _isActive = false;
    }

    [JsonIgnore]
    public bool IsRecording => _writer?.IsRecording ?? false;

    public override void Initialize(UnifiedTrackingData data)
    {
        if (_writer != null)
        {
            return;
        }

        _writer = new RecordingWriter(this, Logger ?? NullLogger.Instance);
        Current = this;

        var modules = UnifiedTracking.Mutator?.LibManager?.LoadedModulesMetadata;
        if (modules != null)
        {
            modules.CollectionChanged += (_, _) => _writer.EnqueueEvent("modules:" + RecordingWriter.DescribeModules());
        }
    }

    // Tap 1: raw module output.
    public override void MutateData(ref UnifiedTrackingData data)
    {
        if (_faulted || _writer == null)
        {
            return;
        }

        try
        {
            _pending ??= ArrayPool<float>.Shared.Rent(RecordingFormat.FrameFloats);
            RecordingFormat.WriteBlock(data, _pending, 0);
            _pendingTick = Stopwatch.GetTimestamp();
        }
        catch (Exception e)
        {
            Fault(e);
        }
    }

    // Tap 2: final output of the same frame.
    public void OnMutationsApplied(UnifiedTrackingData output)
    {
        var frame = _pending;
        if (frame == null || _faulted || _writer == null)
        {
            return;
        }

        try
        {
            _pending = null;
            RecordingFormat.WriteBlock(output, frame, RecordingFormat.BlockLength);
            _writer.EnqueueFrame(_pendingTick, frame);
        }
        catch (Exception e)
        {
            Fault(e);
        }
    }

    [MutationButton("Add marker (F8)")]
    public void AddMarker()
    {
        if (_writer == null || !IsActive || !_writer.IsRecording)
        {
            Logger?.LogWarning("Marker ignored: the Session Recorder isn't recording (it's switched off, no tracking data is arriving, or it paused for disk space; see the log).");
            return;
        }

        var label = string.IsNullOrWhiteSpace(markerLabel) ? "mark" : markerLabel.Trim();
        var number = Interlocked.Increment(ref _markerCount);
        _writer.EnqueueMarker(label);
        Logger?.LogInformation("Recorder marker {Number}: {Label}", number, label);
    }

    /// <summary>
    /// Adds a "kind:detail" event to the recording, e.g. from Adaptive Calibration or the guided calibration, so
    /// offline tools can line them up with the frames. Never blocks; dropped when nothing is being recorded.
    /// </summary>
    public static void LogEvent(string kind, string detail)
    {
        var recorder = Current;
        if (recorder is { _faulted: false, IsActive: true })
        {
            recorder._writer?.EnqueueEvent(kind + ":" + detail);
        }
    }

    [MutationButton("Open recordings folder")]
    public void OpenRecordingsFolder()
    {
        Directory.CreateDirectory(RecordingsDirectory);
        Process.Start(new ProcessStartInfo { FileName = RecordingsDirectory, UseShellExecute = true });
    }

    /// <summary> Closes the current file and stops the writer (also runs on process exit). </summary>
    public void Shutdown(TimeSpan timeout) => _writer?.Shutdown(timeout);

    // A recorder bug must never break tracking: stop recording, keep the pipeline running.
    private void Fault(Exception e)
    {
        _faulted = true;
        Logger?.LogError(e, "Session Recorder failed and stopped recording until VRCFaceTracking restarts.");
    }
}
