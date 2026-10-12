namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// Raw expression frames (module output, before any calibration) for a running guided calibration. Published by
/// the Adaptive Calibration mutation, which runs before the stock Calibration. Kept out of the mutation class on
/// purpose: the settings UI walks every field of a mutation, static ones too (UPSTREAM_BUGS.md #5).
/// </summary>
public static class GuidedTap
{
    public delegate void RawFrameHandler(double t, ReadOnlySpan<float> raw);

    /// <summary> Raised on the tracking thread; handlers must copy what they keep and return quickly. </summary>
    public static event RawFrameHandler? RawFrame;

    public static bool HasListeners => RawFrame != null;

    private static volatile bool _measuring;

    /// <summary>
    /// True while a guided calibration is measuring poses. Adaptive Calibration keeps calibrating then but doesn't learn:
    /// the poses are deliberate maximum efforts, not normal use (set by <see cref="GuidedSession"/>).
    /// </summary>
    public static bool Measuring
    {
        get => _measuring;
        set => _measuring = value;
    }

    public static void Publish(double t, ReadOnlySpan<float> raw) => RawFrame?.Invoke(t, raw);
}
