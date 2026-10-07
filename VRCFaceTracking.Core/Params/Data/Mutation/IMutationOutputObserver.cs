namespace VRCFaceTracking.Core.Params.Data.Mutation;

/// <summary>
/// Adaptive-calibration fork: an active mutation that implements this also sees the final output, after every
/// mutation has run for the frame (e.g. the session recorder's second tap). It must not modify the data.
/// </summary>
public interface IMutationOutputObserver
{
    void OnMutationsApplied(UnifiedTrackingData output);
}
