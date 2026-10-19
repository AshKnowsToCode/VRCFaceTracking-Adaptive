namespace VRCFaceTracking.Core.Adaptive.Calibration;

/// <summary>
/// Adaptive calibration v0 (DESIGN.md section 6, v0 scope in section 10): per expression it learns a rest level and
/// noise (-> soft deadzone) and a ceiling from strong held peaks only, starting from per-module priors. In v0 the
/// ceiling may only rise, at most 1.6x, so normal use can make a shape calmer but never hotter. Shapes defined
/// relative to another (MouthClosed vs JawOpen) take their parent's scale times a fixed ratio.
/// </summary>
/// <remarks>
/// A line-by-line port of tools/replay/adaptive_v0.py, which is the golden reference: tools/replay/golden_v0.py
/// replays recordings through both and requires the outputs to match. Keep the two in step, including the order of
/// floating-point operations. Not thread-safe; one <see cref="Step"/> per module frame. Doubles inside.
/// </remarks>
public sealed class AdaptiveCalibrator
{
    private const int RingSize = 64;

    // Occlusion gate (CalibrationParams.OcclusionLevel): expressions that pull against each other, so both sides high at
    // once is no face but something in front of the mouth cameras (drinking on 2026-10-17). Each group is averaged. Same
    // pairs and order as OCCLUSION_PAIRS in adaptive_v0.py, where the choice is explained.
    private static readonly string[] Pucker = ["LipPuckerUpperLeft", "LipPuckerUpperRight", "LipPuckerLowerLeft", "LipPuckerLowerRight"];
    private static readonly string[] Funnel = ["LipFunnelUpperLeft", "LipFunnelUpperRight", "LipFunnelLowerLeft", "LipFunnelLowerRight"];
    private static readonly string[] Suck = ["LipSuckUpperLeft", "LipSuckUpperRight", "LipSuckLowerLeft", "LipSuckLowerRight"];
    private static readonly string[] Smile = ["MouthCornerPullLeft", "MouthCornerPullRight"];
    private static readonly string[] Stretch = ["MouthStretchLeft", "MouthStretchRight"];
    private static readonly (string[] A, string[] B)[] OcclusionPairs =
    [
        (Smile, Pucker), (Smile, Funnel), (Smile, ["MouthFrownLeft", "MouthFrownRight"]),
        (Stretch, Pucker), (Stretch, Funnel),
        (["JawOpen"], ["MouthPressLeft", "MouthPressRight"]),
        (["MouthRaiserLower"], ["MouthLowerDownLeft", "MouthLowerDownRight"]),
        (Suck, Pucker), (Suck, Funnel),
        (["CheekPuffLeft", "CheekPuffRight"], ["CheekSuckLeft", "CheekSuckRight"]),
        (["MouthUpperLeft"], ["MouthUpperRight"]), (["JawLeft"], ["JawRight"]),
        // not antagonists, but never seen together outside drinking on nov-ash's Quest Pro (sipping); re-check with others
        (Pucker, ["MouthRaiserUpper", "MouthRaiserLower"]),
    ];

    private readonly CalibrationParams _p;
    private readonly int _n;
    private readonly int _bins;
    private readonly int _keep;

    private readonly double[] _c0;
    private readonly double[] _cLo;
    private readonly double[] _cHi;
    private readonly double[] _b0;
    private readonly double[] _sigma0;
    private readonly (int Child, int Parent, double Ratio)[] _coupled;
    private readonly (int[] A, int[] B)[] _occlPairs;
    private double _occlUntil = double.NegativeInfinity;

    // learned state (persisted)
    private double _activeS;
    private readonly double[] _c;
    private readonly double[] _g;
    private readonly double[] _b;
    private readonly double[] _sigma;
    private readonly double[] _hist; // [_n * _bins]
    private double _histN; // decayed sample count, the same for every shape
    private readonly double[] _peakV; // [_n * _keep]
    private readonly double[] _peakT;
    private readonly int[] _peakN;

    // runtime state (not persisted)
    private double[]? _prevX;
    private double _prevT;
    private double _lastChangeT;
    private double _pauseUntil = double.NegativeInfinity;
    private bool _wasBad;
    private readonly double[] _ringT = new double[RingSize];
    private readonly double[] _ringX; // [RingSize * _n]
    private int _ringI;
    private double _histAcc;
    private double _updateAcc;
    private double _guardAcc;
    private readonly bool[] _epOn;
    private readonly double[] _epT0;
    private readonly double[] _epPeak;
    private readonly bool[] _epBad;
    private readonly double[] _satRun; // current pinned run length, active seconds
    private readonly double[] _satShort; // decayed seconds spent in short pinned runs
    private readonly double[] _aboveAcc; // decayed seconds busy (shown >= GuardBusyLevel, or above theta)
    private readonly double[] _satNew; // seconds in short pinned runs since this shape's last guardrail step
    private readonly double[] _pinMin; // [_n * _pinSlots] active minute of the latest short pin per slot (minute % slots)
    private readonly int _pinSlots;

    // scratch
    private readonly double[] _x;
    private readonly double[] _h;
    private readonly double[] _y;
    private readonly double[] _den;
    private readonly bool[] _above;

    public AdaptiveCalibrator(CalibrationPriors priors, CalibrationParams? parameters = null)
    {
        _p = parameters ?? new CalibrationParams();
        Priors = priors;
        _n = priors.Expressions.Length;
        _bins = _p.HistBins;
        _keep = _p.PeaksKeep;

        _c0 = new double[_n];
        _cLo = new double[_n];
        _cHi = new double[_n];
        for (var i = 0; i < _n; i++)
        {
            _c0[i] = Math.Min(Math.Max(priors.C0[i], _p.CAbsMin), 1.0);
            _cLo[i] = Math.Max(_c0[i] / _p.BoundDown, _p.CAbsMin);
            var cMax = priors.CMax?[i] ?? 0.0;
            _cHi[i] = Math.Min(Math.Max(_c0[i] * _p.BoundUp, cMax), 1.0);
        }

        _b0 = (double[])priors.B0.Clone();
        _sigma0 = (double[])priors.Sigma0.Clone();

        _c = (double[])_c0.Clone();
        _g = Enumerable.Repeat(1.0, _n).ToArray();
        _b = (double[])priors.B0.Clone();
        _sigma = priors.Sigma0.Select(s => Math.Min(Math.Max(s, _p.SigmaFloor), _p.SigmaMax)).ToArray();
        _hist = new double[_n * _bins];
        _peakV = new double[_n * _keep];
        _peakT = new double[_n * _keep];
        _peakN = new int[_n];

        Array.Fill(_ringT, double.NegativeInfinity);
        _ringX = new double[RingSize * _n];
        _epOn = new bool[_n];
        _epT0 = new double[_n];
        _epPeak = new double[_n];
        _epBad = new bool[_n];
        _satRun = new double[_n];
        _satShort = new double[_n];
        _aboveAcc = new double[_n];
        _satNew = new double[_n];
        _pinSlots = Math.Max(_p.GuardMinutes, 1);
        _pinMin = new double[_n * _pinSlots];
        Array.Fill(_pinMin, -1e9);

        _x = new double[_n];
        _h = new double[_n];
        _y = new double[_n];
        _den = new double[_n];
        _above = new bool[_n];

        var index = new Dictionary<string, int>();
        for (var i = 0; i < _n; i++)
        {
            index[priors.Expressions[i]] = i;
        }

        _coupled = priors.Coupled
            .Where(kv => index.ContainsKey(kv.Key) && index.ContainsKey(kv.Value.Parent))
            .Select(kv => (index[kv.Key], index[kv.Value.Parent], kv.Value.Ratio))
            .ToArray();

        _occlPairs = OcclusionPairs
            .Where(pair => pair.A.Concat(pair.B).All(index.ContainsKey))
            .Select(pair => (pair.A.Select(s => index[s]).ToArray(), pair.B.Select(s => index[s]).ToArray()))
            .ToArray();
    }

    public CalibrationPriors Priors { get; }

    public CalibrationParams Params => _p;

    public int Count => _n;

    /// <summary> Seconds of usable tracking data learned from (the gated clock). </summary>
    public double ActiveSeconds => _activeS;

    /// <summary> True when the last frame was not used for learning (zeros, frozen, jumps, recovering). </summary>
    public bool Gated { get; private set; }

    /// <summary> When false, frames are mapped with the current state but nothing is learned. </summary>
    public bool LearningEnabled { get; set; } = true;

    /// <summary> The last frame looked occluded (opposing expressions both high). </summary>
    public bool Occluded { get; private set; }

    /// <summary> Occlusion pauses started since this calibrator was created (diagnostics, not saved). </summary>
    public int OcclusionMoments { get; private set; }

    /// <summary> Seconds of otherwise usable data the occlusion gate kept out of learning (diagnostics, not saved). </summary>
    public double OcclusionPausedSeconds { get; private set; }

    /// <summary> Current ceiling divided by its prior, per expression (1 = unchanged, above 1 = calmer). </summary>
    public double CeilingRatio(int i) => _c[i] * _g[i] / _c0[i];

    public double Ceiling(int i) => _c[i];

    public double Guard(int i) => _g[i];

    public double Rest(int i) => _b[i];

    public double Noise(int i) => _sigma[i];

    /// <summary> Highest held peak (raw) among the ones kept for this expression, 0 when none (everyday evidence). </summary>
    public double MaxHeldPeak(int i)
    {
        var m = 0.0;
        for (var k = 0; k < _peakN[i]; k++)
        {
            m = Math.Max(m, _peakV[i * _keep + k]);
        }

        return m;
    }

    /// <summary> Calibrates one frame. <paramref name="t"/> in seconds on any monotonic clock. </summary>
    public void Step(double t, ReadOnlySpan<float> input, Span<float> output)
    {
        Step(t, input);
        for (var i = 0; i < _n; i++)
        {
            output[i] = (float)_y[i];
        }
    }

    /// <summary> Like <see cref="Step(double, ReadOnlySpan{float}, Span{float})"/>, output in double precision. </summary>
    public void Step(double t, ReadOnlySpan<float> input, Span<double> output)
    {
        Step(t, input);
        _y.AsSpan().CopyTo(output);
    }

    private void Step(double t, ReadOnlySpan<float> input)
    {
        var p = _p;
        var n = _n;

        // clip(nan_to_num(x), 0, 1)
        var nanFrame = false;
        for (var i = 0; i < n; i++)
        {
            double v = input[i];
            if (double.IsNaN(v))
            {
                nanFrame = true;
                v = 0.0;
            }

            _x[i] = v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
        }

        // held signal (rolling min over HoldS); needs no gating, and the occlusion check below uses it
        _ringT[_ringI] = t;
        Array.Copy(_x, 0, _ringX, _ringI * n, n);
        _ringI = (_ringI + 1) % RingSize;
        var cut = t - p.HoldS;
        Array.Fill(_h, double.PositiveInfinity);
        for (var r = 0; r < RingSize; r++)
        {
            if (_ringT[r] >= cut)
            {
                var row = r * n;
                for (var i = 0; i < n; i++)
                {
                    if (_ringX[row + i] < _h[i])
                    {
                        _h[i] = _ringX[row + i];
                    }
                }
            }
        }

        // 6.1 gating
        if (_prevX == null)
        {
            _prevX = (double[])_x.Clone();
            _prevT = t;
            _lastChangeT = t;
        }

        double dxMax = 0.0;
        var jumps = 0;
        var anyNonZero = false;
        for (var i = 0; i < n; i++)
        {
            var dx = Math.Abs(_x[i] - _prevX[i]);
            if (dx > dxMax)
            {
                dxMax = dx;
            }

            if (dx > p.JumpSize)
            {
                jumps++;
            }

            if (_x[i] != 0.0)
            {
                anyNonZero = true;
            }
        }

        if (dxMax > p.FrozenEps)
        {
            _lastChangeT = t;
        }

        var zeros = !anyNonZero;
        var frozen = (t - _lastChangeT) >= p.FrozenS;
        if (jumps >= p.JumpCount)
        {
            _pauseUntil = Math.Max(_pauseUntil, t + p.JumpPauseS);
        }

        var bad = zeros || frozen || nanFrame;
        if (_wasBad && !bad)
        {
            _pauseUntil = Math.Max(_pauseUntil, t + p.RecoverS);
        }

        _wasBad = bad;
        Occluded = false;
        if (p.OcclusionLevel > 0.0)
        {
            foreach (var (a, b) in _occlPairs)
            {
                if (Math.Min(Mean(_h, a), Mean(_h, b)) >= p.OcclusionLevel)
                {
                    Occluded = true;
                    if (t >= _occlUntil)
                    {
                        OcclusionMoments++;
                    }

                    _occlUntil = Math.Max(_occlUntil, t + p.OcclusionPauseS);
                    break;
                }
            }
        }

        var gated = bad || t < _pauseUntil || t < _occlUntil;
        Gated = gated;
        var dt = Math.Min(Math.Max(t - _prevT, 0.0), p.MaxFrameDt);
        if (!bad && t >= _pauseUntil && t < _occlUntil)
        {
            OcclusionPausedSeconds += dt;
        }

        Array.Copy(_x, _prevX, n);
        _prevT = t;
        var learn = !gated && LearningEnabled;
        var adt = learn ? dt : 0.0;
        _activeS += adt;

        if (learn)
        {
            // 6.2 rest histogram, HistHz on the active clock
            _histAcc += adt;
            if (_histAcc >= 1.0 / p.HistHz)
            {
                _histAcc -= 1.0 / p.HistHz;
                var decay = Math.Pow(2.0, -1.0 / (p.HistHz * p.HistHalfLifeS));
                for (var k = 0; k < _hist.Length; k++)
                {
                    _hist[k] *= decay;
                }

                _histN = _histN * decay + 1.0;
                for (var i = 0; i < n; i++)
                {
                    var bin = Math.Min((int)(_x[i] * _bins), _bins - 1);
                    _hist[i * _bins + bin] += 1.0;
                }
            }
        }

        // 6.3 episodes on the held signal
        for (var i = 0; i < n; i++)
        {
            var theta = _b[i] + p.ThetaK * _sigma[i] + p.ThetaAdd;
            var above = _h[i] > theta;
            if (above && !_epOn[i])
            {
                _epT0[i] = t;
                _epPeak[i] = _h[i];
                _epBad[i] = !learn;
            }
            else if (above)
            {
                _epPeak[i] = Math.Max(_epPeak[i], _h[i]);
                if (!learn)
                {
                    _epBad[i] = true;
                }
            }
            else if (_epOn[i] && !_epBad[i] && (t - _epT0[i]) >= p.EpisodeMinS)
            {
                AddPeak(i, _epPeak[i]);
            }

            _epOn[i] = above;
            _above[i] = above;
        }

        // output with the current state
        Map();

        // 6.4 guardrail bookkeeping
        if (p.Guardrail && learn)
        {
            var w = Math.Pow(2.0, -adt / p.GuardWindowS);
            var minute = Math.Floor(_activeS / 60.0);
            var slot = (int)(minute % _pinSlots);
            for (var i = 0; i < n; i++)
            {
                var pinned = _y[i] >= p.SatLevel;
                var shortRun = !pinned && _satRun[i] > 0.0 && _satRun[i] < p.SatRunMaxS;
                _satShort[i] = _satShort[i] * w + (shortRun ? _satRun[i] : 0.0);
                _satNew[i] = _satNew[i] + (shortRun ? _satRun[i] : 0.0);
                if (shortRun)
                {
                    _pinMin[i * _pinSlots + slot] = minute;
                }

                _satRun[i] = pinned ? _satRun[i] + adt : 0.0;
                var busy = p.GuardBusyLevel <= 0.0 ? _above[i] : _y[i] >= p.GuardBusyLevel;
                _aboveAcc[i] = _aboveAcc[i] * w + (busy ? adt : 0.0);
            }
        }

        // periodic learning
        if (learn)
        {
            _updateAcc += adt;
            if (_updateAcc >= p.UpdateEveryS)
            {
                Update(_updateAcc);
                _updateAcc = 0.0;
            }
        }
    }

    // Same order of additions as numpy's mean on these short groups (sequential, then one division).
    private static double Mean(double[] v, int[] idx)
    {
        var s = 0.0;
        foreach (var i in idx)
        {
            s += v[i];
        }

        return s / idx.Length;
    }

    private void Map()
    {
        var p = _p;
        for (var i = 0; i < _n; i++)
        {
            var d = p.DeadzoneK * _sigma[i];
            var dd = Math.Max(d, 1e-9);
            var u = _x[i] - _b[i];
            var z = u <= 0.0 ? 0.0 : u < 2 * dd ? u * u / (4 * dd) : u - dd; // C1 soft knee
            var den = Math.Max(_c[i] * _g[i] - _b[i] - d, p.MinDenominator);
            _den[i] = den;
            var y = z / den;
            _y[i] = y < 0.0 ? 0.0 : y > 1.0 ? 1.0 : y;
        }

        foreach (var (child, parent, ratio) in _coupled)
        {
            _y[child] = Math.Min(Math.Max(_x[child] * ratio / _den[parent], 0.0), 1.0);
        }
    }

    private void AddPeak(int i, double v)
    {
        var k = _peakN[i];
        var row = i * _keep;
        if (k < _keep)
        {
            _peakV[row + k] = v;
            _peakT[row + k] = _activeS;
            _peakN[i] = k + 1;
            return;
        }

        // replace the weakest (age-weighted) peak if the new one beats it
        var j = 0;
        var best = double.PositiveInfinity;
        var wj = 0.0;
        for (var m = 0; m < _keep; m++)
        {
            var w = Math.Pow(2.0, -(_activeS - _peakT[row + m]) / _p.PeakHalfLifeS);
            var s = _peakV[row + m] * w;
            if (s < best)
            {
                best = s;
                j = m;
                wj = w;
            }
        }

        if (v > _peakV[row + j] * wj)
        {
            _peakV[row + j] = v;
            _peakT[row + j] = _activeS;
        }
    }

    private void Update(double elapsed)
    {
        var p = _p;

        // 6.2 rest + noise from the histogram
        if (_histN >= p.RestMinSamples)
        {
            var width = 1.0 / _bins;
            for (var i = 0; i < _n; i++)
            {
                var q05 = Quantile(i, p.RestQ);
                var q25 = Quantile(i, 0.25);
                var q50 = Quantile(i, 0.5);
                var ok = (q50 - q05) <= p.RestMaxSpread;
                var sig = LowerHalfMad(i, q25, q50, width) * 1.4826;
                _b[i] = Math.Min(ok ? q05 : _b0[i], _b0[i] + p.RestMaxRise);
                _sigma[i] = Math.Min(Math.Max(ok ? sig : _sigma0[i], p.SigmaFloor), p.SigmaMax);
            }
        }

        // 6.3 ceilings
        var slewUp = Math.Log(1.0 + p.SlewUpPerMin) * elapsed / 60.0;
        var slewDown = Math.Log(1.0 + p.SlewPer10Min) * elapsed / 600.0;
        Span<double> v = stackalloc double[_keep];
        Span<double> w = stackalloc double[_keep];
        Span<int> order = stackalloc int[_keep];
        for (var i = 0; i < _n; i++)
        {
            var k = _peakN[i];
            if (k == 0)
            {
                continue;
            }

            var row = i * _keep;
            for (var m = 0; m < k; m++)
            {
                v[m] = _peakV[row + m];
                w[m] = Math.Pow(2.0, -(_activeS - _peakT[row + m]) / p.PeakHalfLifeS);
                order[m] = m;
            }

            // descending by value, ties by index (stable, like numpy's argsort on <= 16 elements)
            for (var a = 1; a < k; a++)
            {
                var o = order[a];
                var bIdx = a - 1;
                while (bIdx >= 0 && v[order[bIdx]] < v[o])
                {
                    order[bIdx + 1] = order[bIdx];
                    bIdx--;
                }

                order[bIdx + 1] = o;
            }

            var cum = 0.0;
            var pick = -1;
            for (var m = 0; m < k; m++)
            {
                cum += w[order[m]];
                if (pick < 0 && cum >= p.KWeight)
                {
                    pick = m;
                }
            }

            if (cum < p.KWeight)
            {
                continue; // fewer than ~3 recent held peaks: no evidence either way
            }

            var cObs = v[order[pick]];
            double target;
            if (cObs > _c[i])
            {
                target = cObs; // held expressions go past the ceiling: follow them (calmer, never pinned)
            }
            else
            {
                var nEff = Math.Min(cum, p.NEffMax);
                var logC = (p.N0 * Math.Log(_c0[i]) + nEff * Math.Log(p.Eta * cObs)) / (p.N0 + nEff);
                target = Math.Exp(logC);
            }

            target = Math.Min(Math.Max(target, _cLo[i]), _cHi[i]);
            var step = Math.Min(Math.Max(Math.Log(target / _c[i]), -slewDown), slewUp);
            _c[i] = Math.Min(Math.Max(_c[i] * Math.Exp(step), _cLo[i]), _cHi[i]);
        }

        // 6.4 guardrail
        if (p.Guardrail)
        {
            _guardAcc += elapsed;
            var relax = Math.Pow(2.0, -elapsed / p.GuardHalfLifeS);
            for (var i = 0; i < _n; i++)
            {
                _g[i] = 1.0 + (_g[i] - 1.0) * relax;
            }

            if (_guardAcc >= p.GuardEveryS)
            {
                _guardAcc = 0.0;
                var since = Math.Floor(_activeS / 60.0) - p.GuardMinutes;
                for (var i = 0; i < _n; i++)
                {
                    var share = _satShort[i] / Math.Max(_aboveAcc[i], 1e-9);
                    var hot = share > p.GuardShare && _aboveAcc[i] > 10.0;
                    if (p.GuardNeedsNewPins)
                    {
                        hot &= _satNew[i] > 0.0;
                    }

                    if (p.GuardMinMinutes > 0)
                    {
                        // slots holding one of the last GuardMinutes active minutes = minutes with short pins
                        var recent = 0;
                        for (var s = 0; s < _pinSlots; s++)
                        {
                            if (_pinMin[i * _pinSlots + s] > since)
                            {
                                recent++;
                            }
                        }

                        hot &= recent >= p.GuardMinMinutes;
                    }

                    if (hot)
                    {
                        _g[i] *= p.GuardStep;
                        _satNew[i] = 0.0;
                    }

                    _g[i] = Math.Min(_g[i], _cHi[i] / _c[i]);
                }
            }
        }
    }

    /// <summary> Centre of the first histogram bin whose cumulative count reaches q of the total. </summary>
    private double Quantile(int i, double q)
    {
        var row = i * _bins;
        var total = 0.0;
        for (var k = 0; k < _bins; k++)
        {
            total += _hist[row + k];
        }

        var threshold = q * total;
        var cdf = 0.0;
        for (var k = 0; k < _bins; k++)
        {
            cdf += _hist[row + k];
            if (cdf >= threshold)
            {
                return (k + 0.5) * (1.0 / _bins);
            }
        }

        return (_bins - 0.5) * (1.0 / _bins);
    }

    /// <summary> Weighted median of |bin centre - q25| over the bins at or below q50. </summary>
    private double LowerHalfMad(int i, double q25, double q50, double width)
    {
        var row = i * _bins;
        Span<double> dist = stackalloc double[_bins];
        Span<double> wts = stackalloc double[_bins];
        Span<int> order = stackalloc int[_bins];
        var m = 0;
        var sum = 0.0;
        for (var k = 0; k < _bins; k++)
        {
            var centre = (k + 0.5) * width;
            if (centre <= q50 + 1e-12)
            {
                dist[m] = Math.Abs(centre - q25);
                wts[m] = _hist[row + k];
                sum += wts[m];
                order[m] = m;
                m++;
            }
        }

        if (sum <= 0.0)
        {
            return 0.0;
        }

        // ascending by distance, ties by bin (stable)
        var keys = dist[..m].ToArray();
        var idx = order[..m].ToArray();
        Array.Sort(keys, idx);
        StableTies(keys, idx);

        var cw = new double[m];
        var acc = 0.0;
        for (var j = 0; j < m; j++)
        {
            acc += wts[idx[j]];
            cw[j] = acc;
        }

        var half = 0.5 * cw[m - 1];
        for (var j = 0; j < m; j++)
        {
            if (cw[j] >= half)
            {
                return keys[j];
            }
        }

        return keys[m - 1];
    }

    // Array.Sort is not stable: put equal keys back in index order.
    private static void StableTies(double[] keys, int[] idx)
    {
        var start = 0;
        for (var j = 1; j <= keys.Length; j++)
        {
            if (j == keys.Length || keys[j] != keys[start])
            {
                if (j - start > 1)
                {
                    Array.Sort(idx, start, j - start);
                }

                start = j;
            }
        }
    }

    /// <summary>
    /// Takes over what another calibrator learned from raw data (rest histogram, noise, held peaks, active time) while
    /// keeping this one's own ceilings (its priors). Used when a kept guided calibration replaces the starting values:
    /// everyday knowledge survives, so held peaks that go past a new, stronger ceiling calm it again right away.
    /// </summary>
    public void CarryOver(AdaptiveCalibrator old)
    {
        if (old._n != _n || old._bins != _bins || old._keep != _keep)
        {
            return;
        }

        _activeS = old._activeS;
        Array.Copy(old._b, _b, _n);
        Array.Copy(old._sigma, _sigma, _n);
        Array.Copy(old._hist, _hist, _hist.Length);
        _histN = old._histN;
        Array.Copy(old._peakV, _peakV, _peakV.Length);
        Array.Copy(old._peakT, _peakT, _peakT.Length);
        Array.Copy(old._peakN, _peakN, _n);
        ClampRest();
    }

    // Rest and noise taken over from elsewhere (saved state, another calibrator) follow this one's limits right away.
    private void ClampRest()
    {
        for (var i = 0; i < _n; i++)
        {
            _b[i] = Math.Min(_b[i], _b0[i] + _p.RestMaxRise);
            _sigma[i] = Math.Min(Math.Max(_sigma[i], _p.SigmaFloor), _p.SigmaMax);
        }
    }

    // ------------------------------------------------------------------------------------------------------ state

    public CalibrationState SaveState() => new()
    {
        Priors = Priors.Name,
        Expressions = Priors.Expressions.ToArray(),
        ActiveSeconds = _activeS,
        C = (double[])_c.Clone(),
        G = (double[])_g.Clone(),
        B = (double[])_b.Clone(),
        Sigma = (double[])_sigma.Clone(),
        HistN = _histN,
        Hist = Rows(_hist, _bins),
        PeakV = Rows(_peakV, _keep),
        PeakT = Rows(_peakT, _keep),
        PeakN = (int[])_peakN.Clone(),
    };

    /// <summary> Restores learned state saved for the same priors and expressions; false (and no change) otherwise. </summary>
    public bool LoadState(CalibrationState s)
    {
        if (s.Schema != CalibrationState.CurrentSchema || s.Priors != Priors.Name ||
            !s.Expressions.SequenceEqual(Priors.Expressions) ||
            s.C.Length != _n || s.Hist.Length != _n || s.Hist.Any(r => r.Length != _bins) ||
            s.PeakV.Length != _n || s.PeakV.Any(r => r.Length != _keep))
        {
            return false;
        }

        _activeS = s.ActiveSeconds;
        s.C.CopyTo(_c, 0);
        s.G.CopyTo(_g, 0);
        s.B.CopyTo(_b, 0);
        s.Sigma.CopyTo(_sigma, 0);
        _histN = s.HistN;
        Flatten(s.Hist, _hist);
        Flatten(s.PeakV, _peakV);
        Flatten(s.PeakT, _peakT);
        s.PeakN.CopyTo(_peakN, 0);
        ClampRest(); // e.g. state saved before SigmaMax existed
        return true;
    }

    private double[][] Rows(double[] flat, int width)
    {
        var rows = new double[_n][];
        for (var i = 0; i < _n; i++)
        {
            rows[i] = flat.AsSpan(i * width, width).ToArray();
        }

        return rows;
    }

    private static void Flatten(double[][] rows, double[] flat)
    {
        for (var i = 0; i < rows.Length; i++)
        {
            rows[i].CopyTo(flat, i * rows[i].Length);
        }
    }
}

/// <summary> The learned (persisted) part of an <see cref="AdaptiveCalibrator"/>, same fields as adaptive_v0.state(). </summary>
public sealed class CalibrationState
{
    public const int CurrentSchema = 1;

    public int Schema { get; set; } = CurrentSchema;
    public string Priors { get; set; } = "";
    public string[] Expressions { get; set; } = [];
    public DateTime SavedUtc { get; set; }
    public double ActiveSeconds { get; set; }
    public double[] C { get; set; } = [];
    public double[] G { get; set; } = [];
    public double[] B { get; set; } = [];
    public double[] Sigma { get; set; } = [];
    public double HistN { get; set; }
    public double[][] Hist { get; set; } = [];
    public double[][] PeakV { get; set; } = [];
    public double[][] PeakT { get; set; } = [];
    public int[] PeakN { get; set; } = [];
}
