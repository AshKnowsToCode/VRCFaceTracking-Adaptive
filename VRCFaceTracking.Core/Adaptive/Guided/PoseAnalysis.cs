namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary> What one guided run measured: per pose the 0.5 s held level of every expression, rest noise, lips ratio. </summary>
public sealed class RunMeasurement
{
    /// <summary> Pose id -> held level per expression (expression order of the run). Poses with too little data are absent. </summary>
    public Dictionary<string, double[]> Held { get; set; } = new();

    /// <summary> 95th percentile per expression over the relax steps. </summary>
    public double[] RestP95 { get; set; } = [];

    /// <summary> Lips pressed with the jaw open: (MouthClosed / JawOpen, JawOpen), or null. </summary>
    public (double Ratio, double Jaw)? LipsRatio { get; set; }
}

/// <summary> One expression's new starting ceiling and why. </summary>
public sealed record ShapeResult(string Name, double Ceiling, string Basis, int MeasuredRuns, double? Spread,
    double Target, double Other, string OtherPose, double Everyday);

/// <summary>
/// Turns guided runs into starting ceilings, with the rules of tools/replay/poses.py <c>analyze</c> (golden-tested by
/// tools/replay/golden_analysis.py): a ceiling is never set below anything the expression reached in any pose or in
/// everyday use (held 0.5 s), so real expressions don't pin; an expression whose own pose doesn't drive it ("leak",
/// "everyday", "not seen") is never made hotter than the value in effect; runs combine by geometric mean when the
/// expression was measured in at least half of them; sideways pairs share the larger ceiling; the lips-pressed pose
/// sets the MouthClosed / JawOpen coupling ratio.
/// </summary>
public static class PoseAnalysis
{
    public const double LeakFraction = 0.8;
    public const double CouplingMinJaw = 0.3;
    private static readonly string[] BasisOrder = ["measured", "everyday", "leak", "not seen"];

    public static bool Valid(double value, double restP95) => value >= Math.Max(0.04, 3 * restP95 + 0.02);

    private sealed record Evidence(double Target, double Other, string OtherPose, double Everyday, string Basis);

    private static Evidence Evidence_(int i, string shape, IReadOnlyList<(string Id, string[] Targets)> poses,
        RunMeasurement run, double everyday)
    {
        var targetPoses = poses.Where(p => p.Targets.Contains(shape) && run.Held.ContainsKey(p.Id)).Select(p => p.Id)
            .ToHashSet();
        var target = targetPoses.Count > 0 ? targetPoses.Max(id => run.Held[id][i]) : 0.0;
        var other = 0.0;
        var otherPose = "-";
        var first = true;
        foreach (var (id, held) in run.Held)
        {
            if (targetPoses.Contains(id))
            {
                continue;
            }

            // Python max over (value, pose id) tuples: larger value, ties by the larger id
            if (first || held[i] > other || (held[i] == other && string.CompareOrdinal(id, otherPose) > 0))
            {
                other = held[i];
                otherPose = id;
                first = false;
            }
        }

        string basis;
        if (!Valid(target, run.RestP95[i]))
        {
            basis = "not seen";
        }
        else if (target < LeakFraction * other && other >= everyday)
        {
            basis = "leak"; // another pose drives this channel harder than its own pose
        }
        else if (target < LeakFraction * everyday)
        {
            basis = "everyday"; // normal use goes further than the pose did
        }
        else
        {
            basis = "measured";
        }

        return new Evidence(target, other, otherPose, everyday, basis);
    }

    private static double Choose(Evidence ev, double now)
    {
        var seen = Math.Max(Math.Max(ev.Target, ev.Other), ev.Everyday);
        if (ev.Basis == "measured")
        {
            return seen;
        }

        return now > 0 ? Math.Max(now, seen) : Math.Max(seen, ev.Basis == "not seen" ? 1.0 : seen);
    }

    /// <param name="names">Expression order of the measurements.</param>
    /// <param name="poses">The poses of the runs (id, targets), in order.</param>
    /// <param name="runs">One or more guided runs.</param>
    /// <param name="now">Per expression: the ceiling in effect (0 = none).</param>
    /// <param name="everyday">Per expression: the highest everyday held level known (0 = none).</param>
    public static (List<ShapeResult> Shapes, double? CouplingRatio) Analyze(IReadOnlyList<string> names,
        IReadOnlyList<(string Id, string[] Targets)> poses, IReadOnlyList<RunMeasurement> runs, double[] now,
        double[] everyday)
    {
        var index = names.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        var targeted = poses.SelectMany(p => p.Targets).Where(index.ContainsKey).Distinct().ToList();
        var results = new Dictionary<string, ShapeResult>();
        foreach (var shape in targeted)
        {
            var i = index[shape];
            var evs = runs.Select(r => Evidence_(i, shape, poses, r, everyday[i])).ToList();
            var vals = evs.Select(e => Choose(e, now[i])).ToList();
            var measured = vals.Where((v, k) => evs[k].Basis == "measured").ToList();
            double c;
            string basis;
            if (measured.Count > 0 && measured.Count * 2 >= evs.Count)
            {
                c = Math.Exp(measured.Sum(Math.Log) / measured.Count);
                basis = "measured";
            }
            else
            {
                c = vals.Max();
                // the conservative value (max), with the most informative label across runs
                basis = evs.Select(e => e.Basis).OrderBy(b => Array.IndexOf(BasisOrder, b)).First();
            }

            var spread = measured.Count > 1 ? measured.Max() / measured.Min() : (double?)null;
            var e0 = evs[0];
            results[shape] = new ShapeResult(shape, c, basis, measured.Count, spread, e0.Target, e0.Other,
                e0.OtherPose, e0.Everyday);
        }

        // sideways: both directions share the larger ceiling, so a direction mix-up can't make one side hotter
        foreach (var (a, b) in PoseLibrary.SidewaysPairs)
        {
            if (results.TryGetValue(a, out var ra) && results.TryGetValue(b, out var rb))
            {
                var hi = Math.Max(ra.Ceiling, rb.Ceiling);
                results[a] = ra with { Ceiling = hi, Basis = ra.Basis + ", L/R pooled" };
                results[b] = rb with { Ceiling = hi, Basis = rb.Basis + ", L/R pooled" };
            }
        }

        double? coupling = null;
        var good = runs.Where(r => r.LipsRatio is { } lr && lr.Jaw >= CouplingMinJaw).Select(r => r.LipsRatio!.Value.Ratio)
            .ToList();
        if (good.Count > 0)
        {
            var r = Stats.Median(good);
            coupling = Math.Min(Math.Max(1.0 / Math.Max(r, 1e-6), 1.0), 2.0);
        }

        return (targeted.Select(s => results[s]).ToList(), coupling);
    }
}
