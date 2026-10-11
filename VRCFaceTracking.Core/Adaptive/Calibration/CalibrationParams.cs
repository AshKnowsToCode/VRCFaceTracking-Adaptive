namespace VRCFaceTracking.Core.Adaptive.Calibration;

/// <summary>
/// Tunables of <see cref="AdaptiveCalibrator"/>. Defaults and meanings match <c>Params</c> in
/// tools/replay/adaptive_v0.py (the golden reference); change both together. DESIGN.md sections 6 and 10.
/// </summary>
public sealed record CalibrationParams
{
    // 6.1 gating (output is still produced; learning is skipped)
    /// <summary> No expression moved more than this ... </summary>
    public double FrozenEps { get; init; } = 1e-4;
    /// <summary> ... for this long -> frozen (headset off, stream paused). </summary>
    public double FrozenS { get; init; } = 0.5;
    /// <summary> This many expressions ... </summary>
    public int JumpCount { get; init; } = 12;
    /// <summary> ... jumping this much in one frame -> implausible. </summary>
    public double JumpSize { get; init; } = 0.3;
    public double JumpPauseS { get; init; } = 1.0;
    /// <summary> Learning stays off this long after all-zero or frozen data ends. </summary>
    public double RecoverS { get; init; } = 2.0;
    /// <summary> The active clock never advances more than this per frame. </summary>
    public double MaxFrameDt { get; init; } = 0.1;

    // 6.2 rest and noise
    public int HistBins { get; init; } = 200;
    public double HistHz { get; init; } = 20.0;
    /// <summary> 10 active minutes. </summary>
    public double HistHalfLifeS { get; init; } = 600.0;
    public double RestQ { get; init; } = 0.05;
    /// <summary> 60 active seconds at 20 Hz before the data replaces the priors. </summary>
    public double RestMinSamples { get; init; } = 1200;
    /// <summary> Q50 - Q05 above this -> not a rest-dominated shape, keep the priors. </summary>
    public double RestMaxSpread { get; init; } = 0.25;
    public double DeadzoneK { get; init; } = 3.0;
    public double SigmaFloor { get; init; } = 0.002;
    /// <summary> Quantiles, ceilings and the guardrail are recomputed on this active cadence. </summary>
    public double UpdateEveryS { get; init; } = 0.5;

    // 6.3 ceiling
    /// <summary> Held signal = rolling minimum over this window. </summary>
    public double HoldS { get; init; } = 0.15;
    /// <summary> Episode threshold = b + ThetaK * sigma + ThetaAdd. </summary>
    public double ThetaK { get; init; } = 4.0;
    public double ThetaAdd { get; init; } = 0.05;
    public double EpisodeMinS { get; init; } = 0.3;
    public int PeaksKeep { get; init; } = 16;
    /// <summary> 20 active hours. </summary>
    public double PeakHalfLifeS { get; init; } = 20 * 3600.0;
    /// <summary> c_obs = the held peak where the accumulated weight reaches this (about the 3rd highest). </summary>
    public double KWeight { get; init; } = 2.5;
    /// <summary> Headroom: c_target = Eta * c_obs (v0: 1.0, see adaptive_v0.py). </summary>
    public double Eta { get; init; } = 1.0;
    /// <summary> Prior strength, in peaks. </summary>
    public double N0 { get; init; } = 6.0;
    public double NEffMax { get; init; } = 30.0;
    /// <summary>
    /// The ceiling stays in [c0 / BoundDown, c0 * BoundUp]. v0: 1.0, so passive learning can only raise it (calmer);
    /// lowering it (hotter) waits for guided-boost evidence.
    /// </summary>
    public double BoundDown { get; init; } = 1.0;
    public double BoundUp { get; init; } = 1.6;
    /// <summary> The ceiling rises at most this fraction per active minute. </summary>
    public double SlewUpPerMin { get; init; } = 0.50;
    /// <summary> The ceiling falls at most this fraction per 10 active minutes. </summary>
    public double SlewPer10Min { get; init; } = 0.10;
    /// <summary> Never more than ~6.7x gain, whatever the priors say. </summary>
    public double CAbsMin { get; init; } = 0.15;
    public double MinDenominator { get; init; } = 0.05;

    // 6.4 saturation guardrail (only ever makes a shape calmer)
    public bool Guardrail { get; init; } = true;
    public double SatLevel { get; init; } = 0.98;
    /// <summary> Only short pinned runs count (speech, flicker), not deliberate holds. </summary>
    public double SatRunMaxS { get; init; } = 0.5;
    public double GuardWindowS { get; init; } = 600.0;
    public double GuardShare { get; init; } = 0.03;
    public double GuardStep { get; init; } = 1.05;
    public double GuardEveryS { get; init; } = 30.0;
    /// <summary> The guardrail factor relaxes back to 1 over 10 active hours. </summary>
    public double GuardHalfLifeS { get; init; } = 36000.0;
}
