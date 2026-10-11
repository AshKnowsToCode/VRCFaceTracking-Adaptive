namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// Measures one pose (or relax step) from its frames, as tools/replay/poses.py <c>measure</c> does: per expression
/// the highest level held for 0.5 s (only full 0.5 s holds count, so a one-frame spike never does), the frames
/// themselves for relax steps (rest noise), and the lips-pressed MouthClosed / JawOpen ratio. Golden-tested by
/// tools/replay/golden_analysis.py.
/// </summary>
public sealed class PoseMeasure
{
    public const double HeldS = 0.5;
    public const double RatioMinJaw = 0.1; // frames used for the lips ratio: jaw open more than this

    private const double WindowEps = 1e-6;
    // a held level counts once the pose has this much data (0.49 s on a 100 Hz grid = poses.py's first full window)
    private const double FullWindowS = HeldS - 0.015;

    private readonly int _n;
    private readonly int _jaw;
    private readonly int _closed;
    private readonly bool _keepFrames;
    private readonly bool _ratio;
    private readonly List<double> _t = new();
    private readonly List<double[]> _x = new();
    private readonly double[] _held;
    private double _clock;
    private double _start = double.NaN;
    private readonly List<double> _ratios = new();
    private readonly List<double> _jaws = new();
    private readonly List<double[]> _frames = new();

    public PoseMeasure(string poseId, IReadOnlyList<string> names, bool isRelax)
    {
        PoseId = poseId;
        _n = names.Count;
        _held = new double[_n];
        Array.Fill(_held, double.NegativeInfinity);
        var index = names.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        _jaw = index["JawOpen"];
        _closed = index["MouthClosed"];
        _keepFrames = isRelax;
        _ratio = poseId == "lips_pressed_open";
    }

    public string PoseId { get; }

    public int Frames { get; private set; }

    /// <summary> Relax steps: every frame, for the rest percentiles. </summary>
    public IReadOnlyList<double[]> RestFrames => _frames;

    public void Update(double dt, ReadOnlySpan<double> x)
    {
        _clock += dt;
        if (double.IsNaN(_start))
        {
            _start = _clock;
        }

        Frames++;
        var row = x.ToArray();
        _t.Add(_clock);
        _x.Add(row);
        if (_keepFrames)
        {
            _frames.Add(row);
        }

        if (_ratio && row[_jaw] > RatioMinJaw)
        {
            _ratios.Add(row[_closed] / row[_jaw]);
            _jaws.Add(row[_jaw]);
        }

        // drop what no window needs any more
        var drop = 0;
        while (drop < _t.Count && _clock - _t[drop] >= HeldS - WindowEps)
        {
            drop++;
        }

        if (drop > 0)
        {
            _t.RemoveRange(0, drop);
            _x.RemoveRange(0, drop);
        }

        if (_clock - _start < FullWindowS - WindowEps)
        {
            return;
        }

        for (var i = 0; i < _n; i++)
        {
            var m = double.PositiveInfinity;
            foreach (var r in _x)
            {
                if (r[i] < m) m = r[i];
            }

            if (m > _held[i]) _held[i] = m;
        }
    }

    /// <summary> Highest 0.5 s hold per expression; null when the pose had under 0.5 s of frames (poses.py skips it). </summary>
    public double[]? Held(double hz = 100.0) =>
        Frames < HeldS * hz || double.IsNegativeInfinity(_held[0]) ? null : (double[])_held.Clone();

    /// <summary> Lips pressed: (median MouthClosed / JawOpen, median JawOpen) over the jaw-open frames, or null. </summary>
    public (double Ratio, double Jaw)? Ratio(double hz = 100.0) =>
        _ratio && _ratios.Count > HeldS * hz ? (Stats.Median(_ratios), Stats.Median(_jaws)) : null;
}

/// <summary> numpy-compatible order statistics (median, linear percentile). </summary>
public static class Stats
{
    public static double Median(IReadOnlyList<double> values)
    {
        var v = values.ToArray();
        Array.Sort(v);
        var n = v.Length;
        return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
    }

    /// <summary> numpy.percentile(values, q), default 'linear' method (including its two-sided interpolation). </summary>
    public static double Percentile(IReadOnlyList<double> values, double q)
    {
        var v = values.ToArray();
        Array.Sort(v);
        var n = v.Length;
        var index = q / 100.0 * (n - 1);
        var lo = (int)Math.Floor(index);
        var hi = Math.Min(lo + 1, n - 1);
        var t = index - lo;
        var a = v[lo];
        var b = v[hi];
        var diff = b - a;
        return t >= 0.5 ? b - diff * (1.0 - t) : a + diff * t;
    }
}
