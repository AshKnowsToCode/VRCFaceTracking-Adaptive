using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.Extensions.Logging;

namespace VRCFaceTracking.Core.Adaptive.Recording;

/// <summary>
/// Microphone loudness for the session recorder: WASAPI shared-mode capture, reduced on the spot to one RMS level
/// (dBFS) per 10 ms block, stamped with the capture time from WASAPI's QPC position. Samples never leave this class
/// and are never stored. Reopens the device when it disappears or the chosen device changes (e.g. Virtual Desktop
/// connecting after VRCFaceTracking started).
/// </summary>
[SupportedOSPlatform("windows")]
public sealed class MicLevelMonitor : IDisposable
{
    private const int BlockMilliseconds = RecordingFormat.AudioBlockMilliseconds;
    private const float SilenceDb = -120f;
    private const long BufferDuration100Ns = 2_000_000;   // 200 ms shared-mode buffer
    private static readonly TimeSpan DeviceCheckEvery = TimeSpan.FromSeconds(3);
    private static readonly TimeSpan RetryAfter = TimeSpan.FromSeconds(3);

    private readonly string _deviceFilter;
    private readonly Action<long, float> _onLevel;
    private readonly Action<string> _onEvent;
    private readonly ILogger _logger;
    private readonly Thread _thread;
    private volatile bool _stop;

    /// <param name="deviceFilter">Case-insensitive part of the device name; blank = default communications mic.</param>
    /// <param name="onLevel">(QPC tick at the end of the block, dBFS), called on the capture thread.</param>
    /// <param name="onEvent">"audio:..." events for the recording.</param>
    public MicLevelMonitor(string deviceFilter, Action<long, float> onLevel, Action<string> onEvent, ILogger logger)
    {
        _deviceFilter = deviceFilter;
        _onLevel = onLevel;
        _onEvent = onEvent;
        _logger = logger;
        _thread = new Thread(Run) { IsBackground = true, Name = "VRCFT Mic Level" };
        _thread.SetApartmentState(ApartmentState.MTA);
        _thread.Start();
    }

    /// <summary> Device name and format while capturing, otherwise empty. </summary>
    public string DeviceDescription { get; private set; } = "";

    public void Dispose()
    {
        _stop = true;
        _thread.Join(TimeSpan.FromSeconds(1));
    }

    private void Run()
    {
        string? lastProblem = null;
        while (!_stop)
        {
            string? problem;
            try
            {
                problem = Capture();
            }
            catch (Exception e)
            {
                problem = "error " + (e is COMException com ? $"0x{com.HResult:X8}" : e.Message);
            }
            DeviceDescription = "";

            // Report each distinct problem once, not on every retry.
            if (problem != null && problem != lastProblem)
            {
                _onEvent("audio:" + problem);
                _logger.LogWarning("Session Recorder mic level: {Problem}", problem);
            }
            lastProblem = problem;

            for (var waited = TimeSpan.Zero; waited < RetryAfter && !_stop; waited += TimeSpan.FromMilliseconds(100))
            {
                Thread.Sleep(100);
            }
        }
    }

    // Captures until stopped (returns null) or until the device goes away or changes (returns the reason).
    private string? Capture()
    {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumeratorComObject();
        IMMDevice? device = null;
        IAudioClient? client = null;
        IAudioCaptureClient? capture = null;
        var format = IntPtr.Zero;
        try
        {
            device = FindDevice(enumerator, out var deviceId, out var deviceName);
            if (device == null)
            {
                return string.IsNullOrEmpty(_deviceFilter) ? "no_device" : $"no_device matching '{_deviceFilter}'";
            }

            var clientId = typeof(IAudioClient).GUID;
            Check(device.Activate(ref clientId, ClsctxAll, IntPtr.Zero, out var clientObject));
            client = (IAudioClient)clientObject;
            Check(client.GetMixFormat(out format));
            var wave = WaveFormat.Read(format);
            if (wave.SampleKind == SampleKind.Unsupported)
            {
                return $"unsupported_format tag={wave.Tag} bits={wave.BitsPerSample}";
            }

            Check(client.Initialize(SharedMode, 0, BufferDuration100Ns, 0, format, IntPtr.Zero));
            var captureId = typeof(IAudioCaptureClient).GUID;
            Check(client.GetService(ref captureId, out var captureObject));
            capture = (IAudioCaptureClient)captureObject;
            Check(client.Start());

            DeviceDescription = $"{deviceName} ({wave.SampleRate} Hz, {wave.Channels} ch, {wave.SampleKind})";
            _onEvent("audio:device " + DeviceDescription);
            _logger.LogInformation("Session Recorder mic level from {Device}", DeviceDescription);

            var blockFrames = Math.Max(1, wave.SampleRate * BlockMilliseconds / 1000);
            var ticksPerFrame = (double)Stopwatch.Frequency / wave.SampleRate;
            var bytes = Array.Empty<byte>();
            double energy = 0;
            var count = 0;
            var nextDeviceCheck = Stopwatch.GetTimestamp() + (long)(DeviceCheckEvery.TotalSeconds * Stopwatch.Frequency);

            while (!_stop)
            {
                Thread.Sleep(BlockMilliseconds);

                while (true)
                {
                    Check(capture.GetNextPacketSize(out var packetFrames));
                    if (packetFrames == 0)
                    {
                        break;
                    }

                    Check(capture.GetBuffer(out var data, out var frames, out var flags, out _, out var qpc100Ns));
                    try
                    {
                        // WASAPI gives the QPC time of the packet's first frame in 100 ns units.
                        var firstTick = (flags & BufferFlagsTimestampError) != 0 || qpc100Ns == 0
                            ? Stopwatch.GetTimestamp() - (long)(frames * ticksPerFrame)
                            : (long)(qpc100Ns * (Stopwatch.Frequency / 1e7));
                        if ((flags & BufferFlagsDataDiscontinuity) != 0)
                        {
                            energy = 0;
                            count = 0;
                        }

                        var silent = (flags & BufferFlagsSilent) != 0;
                        var length = (int)frames * wave.BlockAlign;
                        if (!silent)
                        {
                            if (bytes.Length < length)
                            {
                                bytes = new byte[length];
                            }
                            Marshal.Copy(data, bytes, 0, length);
                        }

                        for (var i = 0; i < frames; i++)
                        {
                            if (!silent)
                            {
                                energy += wave.FrameEnergy(bytes, i);
                            }

                            if (++count == blockFrames)
                            {
                                var rms = Math.Sqrt(energy / count);
                                var db = rms > 1e-6 ? (float)(20 * Math.Log10(rms)) : SilenceDb;
                                _onLevel(firstTick + (long)((i + 1) * ticksPerFrame), db);
                                energy = 0;
                                count = 0;
                            }
                        }
                    }
                    finally
                    {
                        capture.ReleaseBuffer(frames);
                    }
                }

                if (Stopwatch.GetTimestamp() >= nextDeviceCheck)
                {
                    nextDeviceCheck = Stopwatch.GetTimestamp() + (long)(DeviceCheckEvery.TotalSeconds * Stopwatch.Frequency);
                    var current = FindDevice(enumerator, out var currentId, out _);
                    Release(current);
                    if (currentId != deviceId)
                    {
                        return "device_changed";
                    }
                }
            }

            return null;
        }
        finally
        {
            try
            {
                client?.Stop();
            }
            catch (Exception)
            {
                // Device already gone.
            }

            if (format != IntPtr.Zero)
            {
                Marshal.FreeCoTaskMem(format);
            }
            Release(capture);
            Release(client);
            Release(device);
            Release(enumerator);
        }
    }

    private IMMDevice? FindDevice(IMMDeviceEnumerator enumerator, out string id, out string name)
    {
        id = "";
        name = "";
        if (string.IsNullOrEmpty(_deviceFilter))
        {
            if (enumerator.GetDefaultAudioEndpoint(DataFlowCapture, RoleCommunications, out var device) < 0 || device == null)
            {
                return null;   // E_NOTFOUND: no capture device at all.
            }
            id = DeviceId(device);
            name = FriendlyName(device);
            return device;
        }

        Check(enumerator.EnumAudioEndpoints(DataFlowCapture, DeviceStateActive, out var devices));
        try
        {
            Check(devices.GetCount(out var count));
            for (uint i = 0; i < count; i++)
            {
                Check(devices.Item(i, out var device));
                var deviceName = FriendlyName(device);
                if (deviceName.Contains(_deviceFilter, StringComparison.OrdinalIgnoreCase))
                {
                    id = DeviceId(device);
                    name = deviceName;
                    return device;
                }
                Release(device);
            }
        }
        finally
        {
            Release(devices);
        }
        return null;
    }

    private static string DeviceId(IMMDevice device) => device.GetId(out var id) >= 0 ? id : "";

    private static string FriendlyName(IMMDevice device)
    {
        if (device.OpenPropertyStore(StgmRead, out var store) < 0 || store == null)
        {
            return "?";
        }

        try
        {
            var key = DeviceFriendlyNameKey;
            if (store.GetValue(ref key, out var value) < 0)
            {
                return "?";
            }

            try
            {
                return value.Type == VtLpwstr ? Marshal.PtrToStringUni(value.Pointer) ?? "?" : "?";
            }
            finally
            {
                PropVariantClear(ref value);
            }
        }
        finally
        {
            Release(store);
        }
    }

    private static void Check(int hresult)
    {
        if (hresult < 0)
        {
            Marshal.ThrowExceptionForHR(hresult);
        }
    }

    private static void Release(object? comObject)
    {
        if (comObject != null && Marshal.IsComObject(comObject))
        {
            Marshal.ReleaseComObject(comObject);
        }
    }

    private enum SampleKind { Unsupported, Float32, Int16, Int24, Int32 }

    private readonly record struct WaveFormat(int Tag, int Channels, int SampleRate, int BlockAlign, int BitsPerSample, SampleKind SampleKind)
    {
        private const int TagPcm = 1, TagFloat = 3, TagExtensible = 0xFFFE;
        private static readonly Guid SubtypePcm = new("00000001-0000-0010-8000-00aa00389b71");
        private static readonly Guid SubtypeFloat = new("00000003-0000-0010-8000-00aa00389b71");

        // WAVEFORMATEX, optionally WAVEFORMATEXTENSIBLE (SubFormat at byte 24).
        public static WaveFormat Read(IntPtr pointer)
        {
            int tag = (ushort)Marshal.ReadInt16(pointer, 0);
            int channels = (ushort)Marshal.ReadInt16(pointer, 2);
            var sampleRate = Marshal.ReadInt32(pointer, 4);
            int blockAlign = (ushort)Marshal.ReadInt16(pointer, 12);
            int bits = (ushort)Marshal.ReadInt16(pointer, 14);

            var isFloat = tag == TagFloat;
            var isPcm = tag == TagPcm;
            if (tag == TagExtensible && (ushort)Marshal.ReadInt16(pointer, 16) >= 22)
            {
                var subtype = new byte[16];
                Marshal.Copy(pointer + 24, subtype, 0, 16);
                isFloat = new Guid(subtype) == SubtypeFloat;
                isPcm = new Guid(subtype) == SubtypePcm;
            }

            var container = channels > 0 ? blockAlign / channels : 0;
            var kind = (isFloat, isPcm, container) switch
            {
                (true, _, 4) => SampleKind.Float32,
                (_, true, 2) => SampleKind.Int16,
                (_, true, 3) => SampleKind.Int24,
                (_, true, 4) => SampleKind.Int32,
                _ => SampleKind.Unsupported,
            };
            return new WaveFormat(tag, channels, sampleRate, blockAlign, bits, kind);
        }

        // Mean square over the channels of one frame, full scale = 1.
        public double FrameEnergy(byte[] bytes, int frame)
        {
            var offset = frame * BlockAlign;
            var container = BlockAlign / Channels;
            double sum = 0;
            for (var c = 0; c < Channels; c++, offset += container)
            {
                double sample = SampleKind switch
                {
                    SampleKind.Float32 => BitConverter.ToSingle(bytes, offset),
                    SampleKind.Int16 => BitConverter.ToInt16(bytes, offset) / 32768.0,
                    SampleKind.Int24 => ((bytes[offset] << 8 | bytes[offset + 1] << 16 | bytes[offset + 2] << 24) >> 8) / 8388608.0,
                    SampleKind.Int32 => BitConverter.ToInt32(bytes, offset) / 2147483648.0,
                    _ => 0,
                };
                sum += sample * sample;
            }
            return sum / Channels;
        }
    }

    // --- WASAPI / MMDevice COM interop (only the vtable slots up to the last one used) ---

    private const int DataFlowCapture = 1;          // eCapture
    private const int RoleCommunications = 2;       // eCommunications
    private const int DeviceStateActive = 1;        // DEVICE_STATE_ACTIVE
    private const int ClsctxAll = 0x17;             // CLSCTX_ALL
    private const int SharedMode = 0;               // AUDCLNT_SHAREMODE_SHARED
    private const int StgmRead = 0;
    private const ushort VtLpwstr = 31;
    private const uint BufferFlagsDataDiscontinuity = 0x1, BufferFlagsSilent = 0x2, BufferFlagsTimestampError = 0x4;

    private static readonly PropertyKey DeviceFriendlyNameKey =
        new(new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), 14);   // PKEY_Device_FriendlyName

    [StructLayout(LayoutKind.Sequential)]
    private struct PropertyKey(Guid formatId, int propertyId)
    {
        public Guid FormatId = formatId;
        public int PropertyId = propertyId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PropVariant
    {
        public ushort Type;
        public ushort Reserved1, Reserved2, Reserved3;
        public IntPtr Pointer;
        public IntPtr Pointer2;
    }

    [DllImport("ole32.dll")]
    private static extern int PropVariantClear(ref PropVariant value);

    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    private class MMDeviceEnumeratorComObject
    {
    }

    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDeviceEnumerator
    {
        [PreserveSig] int EnumAudioEndpoints(int dataFlow, int stateMask, out IMMDeviceCollection devices);
        [PreserveSig] int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
    }

    [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDeviceCollection
    {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int Item(uint index, out IMMDevice device);
    }

    [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDevice
    {
        [PreserveSig] int Activate(ref Guid interfaceId, int classContext, IntPtr activationParams, [MarshalAs(UnmanagedType.IUnknown)] out object instance);
        [PreserveSig] int OpenPropertyStore(int access, out IPropertyStore properties);
        [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IPropertyStore
    {
        [PreserveSig] int GetCount(out uint count);
        [PreserveSig] int GetAt(uint index, out PropertyKey key);
        [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
    }

    [ComImport, Guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAudioClient
    {
        [PreserveSig] int Initialize(int shareMode, int streamFlags, long bufferDuration, long periodicity, IntPtr format, IntPtr audioSessionGuid);
        [PreserveSig] int GetBufferSize(out uint frames);
        [PreserveSig] int GetStreamLatency(out long latency);
        [PreserveSig] int GetCurrentPadding(out uint frames);
        [PreserveSig] int IsFormatSupported(int shareMode, IntPtr format, out IntPtr closestMatch);
        [PreserveSig] int GetMixFormat(out IntPtr format);
        [PreserveSig] int GetDevicePeriod(out long defaultPeriod, out long minimumPeriod);
        [PreserveSig] int Start();
        [PreserveSig] int Stop();
        [PreserveSig] int Reset();
        [PreserveSig] int SetEventHandle(IntPtr eventHandle);
        [PreserveSig] int GetService(ref Guid interfaceId, [MarshalAs(UnmanagedType.IUnknown)] out object service);
    }

    [ComImport, Guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAudioCaptureClient
    {
        [PreserveSig] int GetBuffer(out IntPtr data, out uint frames, out uint flags, out ulong devicePosition, out ulong qpcPosition);
        [PreserveSig] int ReleaseBuffer(uint frames);
        [PreserveSig] int GetNextPacketSize(out uint frames);
    }
}
