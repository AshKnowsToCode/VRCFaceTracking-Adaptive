using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Expressions;

namespace VRCFaceTracking.Core.Adaptive.Recording;

/// <summary>
/// The .vftrec session recording format, read by tools/replay/vftrec.py. Bump <see cref="SchemaVersion"/> on any
/// change to the layout below.
/// </summary>
/// <remarks>
/// A file is one UTF-8 JSON header line ending in '\n', then a raw deflate stream (RFC 1951) of records. Every record
/// is a type byte, an int64 tick (<see cref="System.Diagnostics.Stopwatch"/>, i.e. QPC) and a payload, little-endian:
/// <list type="bullet">
/// <item>F frame: float32[2 * BlockLength], the raw block (module output before any mutation) then the out block
/// (after all mutations). A block is <see cref="Channels"/> followed by every expression in enum order.</item>
/// <item>A audio: float32 RMS level in dBFS of one 10 ms capture block; the tick is the block's end.</item>
/// <item>M marker: uint16 byte count, UTF-8 label.</item>
/// <item>G gate/event: uint16 byte count, UTF-8 "kind:detail" (tracking, focus, modules, mutations, audio, recorder,
/// adaptive, guided).</item>
/// </list>
/// The writer sync-flushes every few seconds, so a killed process leaves a readable file cut off mid-record.
/// </remarks>
public static class RecordingFormat
{
    public const int SchemaVersion = 1;
    public const string FileExtension = ".vftrec";

    public const byte FrameRecord = (byte)'F';
    public const byte AudioRecord = (byte)'A';
    public const byte MarkerRecord = (byte)'M';
    public const byte GateRecord = (byte)'G';

    /// <summary> Length of the mic block behind one audio record. </summary>
    public const int AudioBlockMilliseconds = 10;

    /// <summary> Eye and head values at the start of each block. </summary>
    public static readonly string[] Channels =
    {
        "eye_left_gaze_x", "eye_left_gaze_y", "eye_left_openness", "eye_left_pupil_mm",
        "eye_right_gaze_x", "eye_right_gaze_y", "eye_right_openness", "eye_right_pupil_mm",
        "head_yaw", "head_pitch", "head_roll", "head_pos_x", "head_pos_y", "head_pos_z",
    };

    /// <summary> Real expressions only: <see cref="UnifiedTrackingData.Shapes"/> has one spare slot at the end. </summary>
    public static readonly int ShapeCount = (int)UnifiedExpressions.Max;
    public static readonly int BlockLength = Channels.Length + ShapeCount;
    public static readonly int FrameFloats = 2 * BlockLength;

    public static string[] ExpressionNames() =>
        Enumerable.Range(0, ShapeCount).Select(i => ((UnifiedExpressions)i).ToString()).ToArray();

    /// <summary> Copies one block (<see cref="BlockLength"/> floats) of <paramref name="data"/> into a frame. </summary>
    public static void WriteBlock(UnifiedTrackingData data, float[] frame, int offset)
    {
        var o = offset;
        var left = data.Eye.Left;
        var right = data.Eye.Right;
        frame[o++] = left.Gaze.x;
        frame[o++] = left.Gaze.y;
        frame[o++] = left.Openness;
        frame[o++] = left.PupilDiameter_MM;
        frame[o++] = right.Gaze.x;
        frame[o++] = right.Gaze.y;
        frame[o++] = right.Openness;
        frame[o++] = right.PupilDiameter_MM;

        var head = data.Head;
        frame[o++] = head.HeadYaw;
        frame[o++] = head.HeadPitch;
        frame[o++] = head.HeadRoll;
        frame[o++] = head.HeadPosX;
        frame[o++] = head.HeadPosY;
        frame[o++] = head.HeadPosZ;

        for (var i = 0; i < ShapeCount; i++)
        {
            frame[o++] = data.Shapes[i].Weight;
        }
    }
}
