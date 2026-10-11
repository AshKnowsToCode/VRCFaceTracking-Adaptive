namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// The guided calibration's fill ring for one pose (DESIGN.md 8.1). The ring fills only while the asked-for expression
/// is clearly above the person's relaxed face, steady, and not overshadowed by an unrelated expression of the same
/// face region; it fills faster the closer the level is to the reference ("your best in this pose"). Two-sided poses
/// fill a left and a right half separately. While it fills, it records each asked-for shape's best 0.5 s hold.
/// </summary>
/// <remarks>
/// Streaming port of <c>run_step</c> (revised rules) in tools/replay/ring_sim.py, the golden reference:
/// tools/replay/golden_ring.py replays recorded pose sessions through both and requires the same fill, timing,
/// recorded values and asks. Keep the two in step. Windows are by time (0.25 s / 0.5 s), which on ring_sim's 100 Hz
/// grid are exactly its 25 / 50 sample windows. One <see cref="Update"/> per tracking frame; not thread-safe.
/// </remarks>
public sealed class PoseRing
{
    public const double FillS = 1.5; // full-speed fill time
    public const double Full = 0.8; // level / reference: full speed at or above
    public const double Half = 0.5; // half speed at or above, quarter speed below (flagged weak)
    public const double Quarter = 0.25;
    public const double HeldS = 0.5; // recorded value = highest level held this long (poses.py)
    public const double SteadyWindowS = 0.5;
    public const double SteadyRatio = 0.7; // over the last 0.5 s, the lowest level >= 70% of the highest
    public const double TremorHoldS = 0.25; // steadiness on the level held over one tremor cycle (>= 4 Hz)
    public const double Dominance = 1.0; // asked-for level >= this x the strongest unrelated (both / reference)
    public const double LipsJawMin = 0.35; // lips-pressed: jaw at least this x the wide-open jaw from step 1
    public const double AskAfterS = 12.0; // offline: no fill within this much holding -> it would ask

    private const double WindowEps = 1e-6;

    public sealed class HalfState
    {
        internal HalfState(string name, int[] cols)
        {
            Name = name;
            Columns = cols;
            Recorded = new double[cols.Length];
            Analysis = new double[cols.Length];
            Array.Fill(Recorded, double.NegativeInfinity);
            Array.Fill(Analysis, double.NegativeInfinity);
        }

        /// <summary> "" (whole ring), "L" or "R". </summary>
        public string Name { get; }

        /// <summary> Expression indices this half watches. </summary>
        public int[] Columns { get; }

        /// <summary> 0..1. </summary>
        public double Fill { get; internal set; }

        public bool Done { get; internal set; }

        /// <summary> Seconds from the start of the pose until the half filled, or null. </summary>
        public double? FilledS { get; internal set; }

        /// <summary> Best 0.5 s held level / reference over the frames that counted, or null. </summary>
        public double? Level { get; internal set; }

        /// <summary> Per column: best 0.5 s hold (raw) over the frames that counted, until the half filled. </summary>
        public double[] Recorded { get; }

        /// <summary> Per column: best 0.5 s hold over every frame of the pose (poses.py's offline value). </summary>
        public double[] Analysis { get; }

        /// <summary> Whether the last frame counted. </summary>
        public bool Counting { get; internal set; }

        /// <summary> Whether, on the last frame, the asked-for expression was clearly above the relaxed face. </summary>
        public bool Active { get; internal set; }

        /// <summary> Whether, on the last frame, no unrelated expression of the same face region was stronger. </summary>
        public bool Dominant { get; internal set; }

        /// <summary> Level / reference on the last frame. </summary>
        public double CurrentLevel { get; internal set; }

        internal double IncTotal;
        internal double IncQuarter;
        internal int Frames;
        internal int ActiveFrames;
        internal int JawFail;
        internal int WrongFail;
        internal int UnsteadyFail;

        /// <summary> Most of the fill came from below half the reference. </summary>
        public bool Weak => IncQuarter > 0.5 * IncTotal;

        /// <summary> Mean of Recorded / Analysis over the columns that clearly moved, or null. </summary>
        public double? RecordedRatio { get; internal set; }
    }

    private readonly int _n;
    private readonly double[] _tref;
    private readonly double[] _ref;
    private readonly double[] _floor;
    private readonly int[] _unrelated;
    private readonly int _jawCol;
    private readonly double _jawThreshold; // raw; NaN when the pose has no jaw check
    private readonly HalfState[] _halves;
    private readonly int[] _allCols;

    // history: time, per-half level and held level, per-target-column raw value
    private readonly List<double> _histT = new();
    private readonly List<double[]> _histLvl = new(); // per half
    private readonly List<double[]> _histS = new(); // per half: level held over TremorHoldS
    private readonly List<double[]> _histX = new(); // per _allCols
    private double _t;

    /// <param name="names">Expression names, in the order of the frames given to <see cref="Update"/>.</param>
    /// <param name="halves">"" -> shapes, or "L" / "R" -> shapes.</param>
    /// <param name="companions">Expressions known to move with this pose (both sides); never "wrong expression".</param>
    /// <param name="channelRef">Per expression: its ceiling (scales unrelated expressions).</param>
    /// <param name="targetRef">Per expression: the reference for the asked-for shapes (your best in this pose).</param>
    /// <param name="floor">Per expression: clearly above the relaxed face.</param>
    /// <param name="jawOpenRaw">Lips-pressed pose only: the held wide-open jaw from the jaw-open pose (raw), or null
    /// to use the jaw's ceiling.</param>
    public PoseRing(string poseId, IReadOnlyList<string> names, IReadOnlyDictionary<string, string[]> halves,
        IReadOnlySet<string> companions, double[] channelRef, double[] targetRef, double[] floor, double? jawOpenRaw)
    {
        PoseId = poseId;
        _n = names.Count;
        _ref = channelRef;
        _tref = targetRef;
        _floor = floor;
        var index = new Dictionary<string, int>();
        for (var i = 0; i < _n; i++)
        {
            index[names[i]] = i;
        }

        var targets = halves.Values.SelectMany(v => v).ToHashSet();
        var regions = targets.SelectMany(FaceRegions.Of).ToHashSet();
        _unrelated = Enumerable.Range(0, _n)
            .Where(j => !targets.Contains(names[j]) && !companions.Contains(names[j]) &&
                        FaceRegions.Of(names[j]).Overlaps(regions))
            .ToArray();
        _jawCol = index["JawOpen"];
        _jawThreshold = poseId == "lips_pressed_open"
            ? LipsJawMin * (jawOpenRaw is > 0 ? jawOpenRaw.Value : channelRef[_jawCol])
            : double.NaN;
        _halves = halves.Select(kv => new HalfState(kv.Key, kv.Value.Select(s => index[s]).ToArray())).ToArray();
        _allCols = _halves.SelectMany(h => h.Columns).Distinct().ToArray();
    }

    public string PoseId { get; }

    public IReadOnlyList<HalfState> Halves => _halves;

    /// <summary> On the last frame: the strongest unrelated expression (index), or -1. </summary>
    public int TopUnrelated { get; private set; } = -1;

    /// <summary> Every half has filled. </summary>
    public bool Done => _halves.All(h => h.Done);

    /// <summary> Seconds of frames seen. </summary>
    public double Elapsed => _t;

    /// <summary> One tracking frame; <paramref name="dt"/> = seconds since the previous frame (live: clamp it). </summary>
    public void Update(double dt, ReadOnlySpan<double> x)
    {
        _t += dt;
        var t = _t;

        // unrelated: strongest expression of the same face region, above its floor, / its ceiling
        var u = double.NegativeInfinity;
        TopUnrelated = -1;
        foreach (var j in _unrelated)
        {
            var v = x[j] >= _floor[j] ? x[j] / _ref[j] : 0.0;
            if (v > u)
            {
                u = v;
                TopUnrelated = j;
            }
        }

        var jawOk = double.IsNaN(_jawThreshold) || x[_jawCol] >= _jawThreshold;

        // history for the trailing windows
        _histT.Add(t);
        var lvls = new double[_halves.Length];
        var xs = new double[_allCols.Length];
        for (var c = 0; c < _allCols.Length; c++)
        {
            xs[c] = x[_allCols[c]];
        }

        for (var hi = 0; hi < _halves.Length; hi++)
        {
            var h = _halves[hi];
            var sum = 0.0;
            foreach (var c in h.Columns)
            {
                sum += x[c] / _tref[c];
            }

            lvls[hi] = sum / h.Columns.Length;
        }

        _histLvl.Add(lvls);
        _histX.Add(xs);
        var s = new double[_halves.Length];
        _histS.Add(s);
        for (var hi = 0; hi < _halves.Length; hi++)
        {
            s[hi] = WindowMin(_histLvl, hi, t, TremorHoldS);
        }

        for (var hi = 0; hi < _halves.Length; hi++)
        {
            var h = _halves[hi];
            var lvl = lvls[hi];
            h.CurrentLevel = lvl;
            h.Frames++;

            double rawMean = 0.0, floorMean = 0.0;
            foreach (var c in h.Columns)
            {
                rawMean += x[c];
                floorMean += _floor[c];
            }

            var active = rawMean / h.Columns.Length > floorMean / h.Columns.Length;
            var steady = WindowMin(_histS, hi, t, SteadyWindowS) >= SteadyRatio * WindowMax(_histS, hi, t, SteadyWindowS);
            var dominant = lvl >= Dominance * u;
            var ok = active && jawOk && dominant && steady;
            h.Active = active;
            h.Dominant = dominant;

            if (active)
            {
                h.ActiveFrames++;
                if (!jawOk) h.JawFail++;
                else if (!dominant) h.WrongFail++;
                else if (!steady) h.UnsteadyFail++;
            }

            // offline analysis value: best 0.5 s hold of each column over the whole pose
            for (var k = 0; k < h.Columns.Length; k++)
            {
                var held = WindowMinX(Array.IndexOf(_allCols, h.Columns[k]), t, HeldS);
                if (held > h.Analysis[k]) h.Analysis[k] = held;
            }

            h.Counting = ok && !h.Done;
            if (h.Done)
            {
                continue;
            }

            var rate = lvl >= Full ? 1.0 : lvl >= Half ? 0.5 : Quarter;
            var inc = ok ? rate * dt / FillS : 0.0;
            h.IncTotal += inc;
            if (rate == Quarter)
            {
                h.IncQuarter += inc;
            }

            h.Fill += inc;
            if (ok)
            {
                var lvlHeld = WindowMin(_histLvl, hi, t, HeldS);
                if (h.Level == null || lvlHeld > h.Level)
                {
                    h.Level = lvlHeld;
                }

                for (var k = 0; k < h.Columns.Length; k++)
                {
                    var held = WindowMinX(Array.IndexOf(_allCols, h.Columns[k]), t, HeldS);
                    if (held > h.Recorded[k]) h.Recorded[k] = held;
                }
            }

            if (h.Fill >= 1.0)
            {
                h.Done = true;
                h.FilledS = t;
            }
        }

        TrimHistory(t);
    }

    /// <summary> Offline summary for the golden test (ring_sim's open-loop "would it ask" verdict). </summary>
    public (double? NeedS, string? Reason) Summarize(HalfState h, double dt)
    {
        double? need = null;
        if (!h.Done && h.Fill > 0)
        {
            need = h.Frames * dt / h.Fill;
        }

        if (h.Done || (need != null && need <= AskAfterS))
        {
            return (need, null);
        }

        return (need, Reason(h));
    }

    /// <summary>
    /// Why a half isn't filling: "no response" (clearly above the relaxed face on under 20% of its frames), else the
    /// most frequent of "jaw not open enough", "wrong expression", "unsteady" (ring_sim's order; ties to the first).
    /// </summary>
    public static string Reason(HalfState h)
    {
        if (h.ActiveFrames < 0.2 * h.Frames)
        {
            return "no response";
        }

        // ring_sim: max over (jaw, wrong expression, unsteady, weaker than target) in that order; the revised rules
        // never stall below half, so "weaker than target" counts 0.
        var reason = "jaw not open enough";
        var best = h.JawFail;
        if (h.WrongFail > best) { best = h.WrongFail; reason = "wrong expression"; }
        if (h.UnsteadyFail > best) { best = h.UnsteadyFail; reason = "unsteady"; }
        if (0 > best) { reason = "weaker than target"; }
        return reason;
    }

    /// <summary> Fills <see cref="HalfState.RecordedRatio"/> (call when the pose is over). </summary>
    public void Finish()
    {
        foreach (var h in _halves)
        {
            if (!h.Done || h.Level == null)
            {
                continue;
            }

            double sum = 0.0;
            var k = 0;
            for (var c = 0; c < h.Columns.Length; c++)
            {
                if (h.Analysis[c] > _floor[h.Columns[c]])
                {
                    sum += h.Recorded[c] / h.Analysis[c];
                    k++;
                }
            }

            h.RecordedRatio = k > 0 ? sum / k : null;
        }
    }

    // window = the frames within the last w seconds, this one included (on a 100 Hz grid: exactly w * 100 samples)
    private double WindowMin(List<double[]> hist, int i, double t, double w)
    {
        var m = double.PositiveInfinity;
        for (var k = hist.Count - 1; k >= 0 && t - _histT[k] < w - WindowEps; k--)
        {
            if (hist[k][i] < m) m = hist[k][i];
        }

        return m;
    }

    private double WindowMax(List<double[]> hist, int i, double t, double w)
    {
        var m = double.NegativeInfinity;
        for (var k = hist.Count - 1; k >= 0 && t - _histT[k] < w - WindowEps; k--)
        {
            if (hist[k][i] > m) m = hist[k][i];
        }

        return m;
    }

    private double WindowMinX(int c, double t, double w) => WindowMin(_histX, c, t, w);

    private void TrimHistory(double t)
    {
        var drop = 0;
        while (drop < _histT.Count && t - _histT[drop] >= HeldS + 0.1)
        {
            drop++;
        }

        if (drop > 0)
        {
            _histT.RemoveRange(0, drop);
            _histLvl.RemoveRange(0, drop);
            _histS.RemoveRange(0, drop);
            _histX.RemoveRange(0, drop);
        }
    }
}

/// <summary> Which part of the face an expression belongs to, for the "wrong expression" check (ring_sim.region). </summary>
public static class FaceRegions
{
    private static readonly HashSet<string> Upper = new() { "upper" };
    private static readonly HashSet<string> Lower = new() { "lower" };
    private static readonly HashSet<string> Both = new() { "upper", "lower" };

    public static IReadOnlySet<string> Of(string expression) =>
        expression.StartsWith("NoseSneer") || expression.StartsWith("CheekSquint") ? Both
        : expression.StartsWith("Eye") || expression.StartsWith("Brow") ? Upper
        : Lower;

    /// <summary> The same expression on the other side (Left <-> Right). </summary>
    public static string Mirror(string expression) =>
        expression.Replace("Left", "\0").Replace("Right", "Left").Replace("\0", "Right");
}
