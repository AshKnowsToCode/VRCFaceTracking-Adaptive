namespace VRCFaceTracking.Core.Params.Data.Mutation;

/// <summary>
/// Adaptive-calibration fork: a mutation that implements this is called first when VRCFaceTracking tears down (app
/// exit, or "force re-init" in the risky settings), before settings are saved and modules are stopped. A Windows
/// shutdown or sign-out ends the app a few seconds into teardown, and AppDomain.ProcessExit never runs then, so
/// whatever must reach the disk goes here. Keep it quick, and keep working afterwards: after a re-init the app runs on.
/// </summary>
public interface IMutationTeardownHandler
{
    void OnTeardown();
}
