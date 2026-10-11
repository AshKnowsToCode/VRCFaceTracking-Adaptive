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

    public static void Publish(double t, ReadOnlySpan<float> raw) => RawFrame?.Invoke(t, raw);
}
