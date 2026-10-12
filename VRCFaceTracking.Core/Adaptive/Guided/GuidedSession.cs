using System.Globalization;
using VRCFaceTracking.Core.Adaptive.Calibration;
using VRCFaceTracking.Core.Adaptive.Recording;

namespace VRCFaceTracking.Core.Adaptive.Guided;

public enum GuidedPhase
{
    Idle,
    Intro,
    Relax,
    Hold,
    Got,
    Ask,
    Summary,
    Cancelled,
}

/// <summary> Something the page should say or play. </summary>
public sealed record GuidedCue(GuidedCueKind Kind, string? Text = null, double Value = 0);

public enum GuidedCueKind
{
    Speak,
    Chime, // a pose is done
    Tick, // the ring passed another quarter (Value = fill)
}

/// <summary> One half of the ring as the page draws it. </summary>
public sealed record HalfView(string Name, double Fill, bool Done, bool Counting, double Level);

/// <summary> Immutable view of a running guided calibration for the page (polled from the UI thread). </summary>
public sealed record GuidedSnapshot(
    GuidedPhase Phase,
    int PoseIndex,
    int PoseCount,
    Pose? Pose,
    IReadOnlyList<HalfView> Halves,
    string Message,
    string? AlsoMoving,
    bool TrackingOk,
    bool PausedByUser,
    bool Redoing,
    GuidedResult? Result);

/// <summary> How one pose went. </summary>
public sealed class PoseOutcome
{
    public required string Id { get; init; }
    public bool Done { get; set; }
    /// <summary> Done again from the summary ("Redo (optional)"); replaces the first attempt. </summary>
    public bool Redone { get; set; }
    public bool Skipped { get; set; }
    public List<string> Asks { get; } = new();
    public double? FilledS { get; set; }
    public bool Weak { get; set; }
    /// <summary> Per asked-for expression: best 0.5 s hold (raw) while the ring counted ("your best in this pose"). </summary>
    public Dictionary<string, double> Recorded { get; } = new();
}

/// <summary> One expression group as the summary shows it. </summary>
public sealed record SummaryRow(string PoseId, string Title, string Change, string Status, SummaryColor Color);

public enum SummaryColor
{
    Green,
    Yellow,
    Grey,
}

/// <summary> Everything a finished run produced (nothing is applied until the person keeps it). </summary>
public sealed class GuidedResult
{
    public required DateTime Started { get; init; }
    public required string ModuleKey { get; init; }
    public required string[] Names { get; init; }
    public required RunMeasurement Run { get; init; }
    public required List<PoseOutcome> Poses { get; init; }
    public required List<ShapeResult> Shapes { get; init; }
    public required double? Coupling { get; init; }
    public required double[] Before { get; init; }
    public required CalibrationPriors NewPriors { get; init; }
    public required List<SummaryRow> Summary { get; init; }
    public required int RunsCombined { get; init; }
}

/// <summary> Earlier kept runs: their measurements (combined with this run) and "your best in this pose". </summary>
public sealed class GuidedHistory
{
    public static readonly GuidedHistory Empty = new();

    public List<RunMeasurement> Runs { get; } = new();

    public Dictionary<string, Dictionary<string, double>> Best { get; } = new();
}

/// <summary>
/// One guided calibration run (DESIGN.md 8.1): relax, then each pose with its <see cref="PoseRing"/>; the ring fills
/// only on good data, never on a timer. It asks (never skips by itself) when a pose doesn't progress, pauses while
/// tracking is lost, and at the end turns the measurements into new starting values with <see cref="PoseAnalysis"/>.
/// Frames arrive on the tracking thread (<see cref="GuidedTap"/>); the page polls <see cref="Snapshot"/> and calls
/// the commands from the UI thread.
/// </summary>
public sealed class GuidedSession : IDisposable
{
    public const double IntroS = 5.0;
    public const double RelaxS = 3.0;
    public const double GotS = 1.6;
    public const double AskAfterS = 10.0; // earliest ask, from the start of a pose
    public const double StallWindowS = 4.0; // asks when the ring gained less than StallMin over this long
    public const double StallMin = 0.02;
    public const double ResumeProgress = 0.05; // an open question clears once the ring gains this much again
    public const double MaxHoldS = 20.0; // a pose that crawls (e.g. a pumping cheek puff) asks after this long anyway
    public const double FrozenS = 0.5;
    public const double FollowAfter = 3; // the reference follows the person after this many poses
    public const double FollowMin = 0.3;
    public const double LeftAloneS = 60.0; // a question unanswered this long: the person went back to normal use
    private const double MaxDt = 0.1;

    private readonly object _lock = new();
    private readonly GuidedInfo _info;
    private readonly IReadOnlyList<Pose> _poses;
    private readonly GuidedModuleData _data;
    private readonly GuidedHistory _history;
    private readonly string[] _names;
    private readonly int _n;
    private readonly DateTime _started = DateTime.Now;

    private GuidedPhase _phase = GuidedPhase.Idle;
    private int _index = -1;
    private double _phaseTime;
    private double _lastT = double.NaN;
    private double _lastChangeT;
    private double[]? _prevX;
    private bool _trackingOk = true;
    private bool _pausedByUser;
    private string _message = "";
    private string? _alsoMoving;

    private PoseRing? _ring;
    private PoseMeasure? _measure;
    private PoseOutcome? _outcome;
    private double[]? _tref;
    private readonly List<(double T, double Fill)> _fillHistory = new();
    private double _askFill;
    private double _lastAskAt = double.NegativeInfinity;
    private int _ticks;
    private bool _saidOtherWay;

    private double[]? _floor;
    private double _scale = 1.0;
    private readonly List<double> _levels = new();
    private double? _jawRaw;
    private readonly List<double[]> _restFrames = new();
    private readonly RunMeasurement _run = new();
    private readonly List<PoseOutcome> _outcomes = new();
    private GuidedResult? _result;
    private GuidedSnapshot _snapshot;
    private bool _subscribed;
    private bool _redoing;

    public GuidedSession(GuidedInfo info, IReadOnlyList<Pose> poses, GuidedModuleData data, GuidedHistory history)
    {
        _info = info;
        _poses = poses;
        _data = data;
        _history = history;
        _names = info.Priors.Expressions;
        _n = _names.Length;
        _snapshot = MakeSnapshot();
    }

    /// <summary> Raised on the tracking thread (outside the session's lock). </summary>
    public event Action<GuidedCue>? Cue;

    public GuidedSnapshot Snapshot => Volatile.Read(ref _snapshot);

    public IReadOnlyList<Pose> Poses => _poses;

    /// <summary> Starts listening to tracking frames (call once). </summary>
    public void Start()
    {
        var cues = new List<GuidedCue>();
        lock (_lock)
        {
            if (_phase != GuidedPhase.Idle)
            {
                return;
            }

            _phase = GuidedPhase.Intro;
            _phaseTime = 0.0;
            Log($"start {_info.ModuleKey}; {_poses.Count} steps");
            _message = "Get comfortable. Each face: hold it until the ring is full.";
            cues.Add(new GuidedCue(GuidedCueKind.Speak,
                "Face calibration. Make each face at full strength, but comfortably, and hold it until the chime."));
            if (!_subscribed)
            {
                _subscribed = true;
                GuidedTap.RawFrame += OnFrame;
            }

            Publish();
        }

        Raise(cues);
    }

    /// <summary> Feeds one frame directly (replay / tests); live frames come from <see cref="GuidedTap"/>. </summary>
    public void Feed(double t, ReadOnlySpan<float> raw) => OnFrame(t, raw);

    public void Skip() => Command(() =>
    {
        if (_redoing)
        {
            AbortRedo();
        }
        else if (_phase is GuidedPhase.Hold or GuidedPhase.Ask && _outcome != null)
        {
            Log("skip " + _outcome.Id);
            _outcome.Skipped = true;
            LeavePose();
            Next();
        }
    });

    public void Retry() => Command(() =>
    {
        if (_phase is GuidedPhase.Hold or GuidedPhase.Ask && _index >= 0)
        {
            EnterPose(_index, retry: true);
        }
    });

    public void SetPaused(bool paused) => Command(() =>
    {
        if (paused != _pausedByUser)
        {
            Log(paused ? "pause" : "unpause");
        }

        _pausedByUser = paused;
    });

    /// <summary> Stops the run; during a redo it only stops the redo (back to the summary, earlier result kept). </summary>
    public void Cancel()
    {
        var cancelled = false;
        Command(() =>
        {
            if (_redoing)
            {
                AbortRedo();
                return;
            }

            Log("stop");
            _phase = GuidedPhase.Cancelled;
            cancelled = true;
        });
        if (cancelled)
        {
            Dispose();
        }
    }

    /// <summary>
    /// From the summary: does one pose again ("Redo (optional)"). When its ring fills, the new measurement replaces
    /// the first one and the summary is worked out again; skipping or stopping keeps the earlier result.
    /// </summary>
    public void Redo(string poseId) => Command(() =>
    {
        if (_phase != GuidedPhase.Summary)
        {
            return;
        }

        var index = -1;
        for (var i = 0; i < _poses.Count; i++)
        {
            if (_poses[i].Id == poseId && !_poses[i].IsRelax)
            {
                index = i;
            }
        }

        if (index < 0)
        {
            return;
        }

        _redoing = true;
        _outcome = null;
        EnterPose(index, retry: false);
    });

    private void AbortRedo()
    {
        Log("redo stopped " + _poses[_index].Id);
        _redoing = false;
        _ring = null;
        _phase = GuidedPhase.Summary;
        _message = "All done.";
        Say("Kept the earlier result.");
    }

    public void Dispose()
    {
        // under the lock: a frame still in flight must not set Measuring again after this clears it
        lock (_lock)
        {
            if (_subscribed)
            {
                _subscribed = false;
                GuidedTap.RawFrame -= OnFrame;
                GuidedTap.Measuring = false;
            }
        }
    }

    private List<GuidedCue> _pending = new();

    private void Command(Action action)
    {
        List<GuidedCue> cues;
        lock (_lock)
        {
            action();
            Publish();
            cues = _pending;
            _pending = new List<GuidedCue>();
        }

        Raise(cues);
    }

    private void Say(string text) => _pending.Add(new GuidedCue(GuidedCueKind.Speak, text));

    // "guided:..." events in the session recording, so a run can be lined up with the frames offline
    private static void Log(string detail) => SessionRecorder.LogEvent("guided", detail);

    private void Raise(List<GuidedCue> cues)
    {
        foreach (var c in cues)
        {
            Cue?.Invoke(c);
        }
    }

    private void OnFrame(double t, ReadOnlySpan<float> raw)
    {
        List<GuidedCue> cues;
        lock (_lock)
        {
            Step(t, raw);
            Publish();
            cues = _pending;
            _pending = new List<GuidedCue>();
        }

        Raise(cues);
    }

    private void Step(double t, ReadOnlySpan<float> raw)
    {
        if (_phase is GuidedPhase.Idle or GuidedPhase.Summary or GuidedPhase.Cancelled)
        {
            return;
        }

        var dt = double.IsNaN(_lastT) ? 0.0 : Math.Min(Math.Max(t - _lastT, 0.0), MaxDt);
        _lastT = t;
        var x = new double[_n];
        var any = false;
        var changed = _prevX == null;
        for (var i = 0; i < _n; i++)
        {
            double v = raw[i];
            x[i] = double.IsNaN(v) ? 0.0 : Math.Min(Math.Max(v, 0.0), 1.0);
            any |= x[i] != 0.0;
            if (_prevX != null && Math.Abs(x[i] - _prevX[i]) > 1e-4)
            {
                changed = true;
            }
        }

        _prevX = x;
        if (changed)
        {
            _lastChangeT = t;
        }

        var ok = any && t - _lastChangeT < FrozenS;
        if (ok != _trackingOk)
        {
            _trackingOk = ok;
            Say(ok ? "Tracking is back." : "Tracking lost. Paused.");
        }

        if (!ok || _pausedByUser)
        {
            return;
        }

        _phaseTime += dt;
        switch (_phase)
        {
            case GuidedPhase.Intro:
                if (_phaseTime >= IntroS)
                {
                    EnterPose(0, retry: false);
                }

                break;

            case GuidedPhase.Relax:
                _measure!.Update(dt, x);
                if (_phaseTime >= RelaxS)
                {
                    _restFrames.AddRange(_measure.RestFrames);
                    if (_floor == null)
                    {
                        _floor = new double[_n];
                        for (var i = 0; i < _n; i++)
                        {
                            var p95 = Stats.Percentile(_restFrames.Select(r => r[i]).ToList(), 95);
                            _floor[i] = Math.Max(0.04, 3 * p95 + 0.02);
                        }
                    }

                    _outcome!.Done = true;
                    _outcomes.Add(_outcome);
                    Next();
                }

                break;

            case GuidedPhase.Hold:
            case GuidedPhase.Ask:
                HoldFrame(dt, x);
                break;

            case GuidedPhase.Got:
                if (_phaseTime >= GotS)
                {
                    if (_redoing)
                    {
                        _redoing = false;
                        Finish(redo: true);
                    }
                    else
                    {
                        Next();
                    }
                }

                break;
        }
    }

    private void HoldFrame(double dt, double[] x)
    {
        var ring = _ring!;
        _measure!.Update(dt, x);
        ring.Update(dt, x);
        var fill = ring.Halves.Average(h => h.Fill);
        _fillHistory.Add((_phaseTime, fill));
        while (_fillHistory.Count > 0 && _phaseTime - _fillHistory[0].T > StallWindowS)
        {
            _fillHistory.RemoveAt(0);
        }

        var quarter = (int)Math.Floor(Math.Min(fill, 0.999) * 4);
        if (quarter > _ticks)
        {
            _ticks = quarter;
            _pending.Add(new GuidedCue(GuidedCueKind.Tick, Value: fill));
        }

        var waiting = ring.Halves.FirstOrDefault(h => !h.Done);
        _alsoMoving = waiting is { Active: true, Dominant: false } && ring.TopUnrelated >= 0
            ? Pretty(_names[ring.TopUnrelated])
            : null;
        _message = _phase == GuidedPhase.Ask ? _message : Coach(ring);

        var pose = _poses[_index];
        if (pose.Sideways && !_saidOtherWay && ring.Halves.Count(h => h.Done) == 1)
        {
            _saidOtherWay = true;
            Say("Now the other way.");
        }

        if (ring.Done)
        {
            Log(string.Create(CultureInfo.InvariantCulture, $"filled {pose.Id} {ring.Elapsed:0.00}"));
            _outcome!.Done = true;
            _outcome.FilledS = ring.Elapsed;
            _pending.Add(new GuidedCue(GuidedCueKind.Chime));
            Say("Got it. Relax.");
            LeavePose();
            _phase = GuidedPhase.Got;
            _phaseTime = 0.0;
            _message = "Got it. Relax.";
            return;
        }

        var gained = _fillHistory.Count > 0 ? fill - _fillHistory[0].Fill : 0.0;
        var stalled = _phaseTime - _fillHistory[0].T >= StallWindowS - 0.25 && gained < StallMin;
        var crawling = _phaseTime >= MaxHoldS;
        if (_phase == GuidedPhase.Hold && _phaseTime >= AskAfterS && _phaseTime - _lastAskAt >= AskAfterS &&
            (stalled || crawling))
        {
            _lastAskAt = _phaseTime;
            var reason = AskReason(ring, pose);
            Log($"ask {pose.Id} {reason.Key}");
            _outcome!.Asks.Add(reason.Key);
            _phase = GuidedPhase.Ask;
            _askFill = fill;
            _message = reason.Text + " Try again, or skip this one.";
            Say(reason.Text + " Try again, or skip this one.");
        }
        else if (_phase == GuidedPhase.Ask && fill - _askFill >= ResumeProgress)
        {
            Log("resumed " + pose.Id);
            _phase = GuidedPhase.Hold; // doing the face again answers the question
        }
    }

    /// <summary> "LipPuckerUpperLeft" -> "lip pucker upper left". </summary>
    public static string Pretty(string expression) =>
        string.Concat(expression.Select((ch, i) => i > 0 && char.IsUpper(ch) ? " " + char.ToLowerInvariant(ch) : ch.ToString()))
            .ToLowerInvariant();

    private string Coach(PoseRing ring)
    {
        var h = ring.Halves.FirstOrDefault(h => !h.Done) ?? ring.Halves[0];
        if (h.Counting)
        {
            return h.CurrentLevel >= PoseRing.Half ? "Good. Hold it." : "Good. Hold it there.";
        }

        return _poses[_index].Prompt;
    }

    private (string Key, string Text) AskReason(PoseRing ring, Pose pose)
    {
        var stuck = ring.Halves.Where(h => !h.Done).ToList();
        var side = stuck.Count == 1 && stuck[0].Name is "L" or "R" ? stuck[0].Name == "L" ? "left" : "right" : null;
        var h = stuck.OrderBy(s => s.Fill).First();
        return PoseRing.Reason(h) switch
        {
            "no response" => ("no response", side != null ? $"Your {side} side isn't registering." : "That isn't registering."),
            "wrong expression" => ("wrong expression", pose.Mistake.Length > 0 ? pose.Mistake : "That looks like a different face."),
            "jaw not open enough" => ("jaw not open enough", "Open your jaw a bit more, and keep the lips sealed."),
            _ => ("unsteady", "Hold it steady."),
        };
    }

    private void Next() => EnterPose(_index + 1, retry: false);

    private void EnterPose(int index, bool retry)
    {
        if (index >= _poses.Count)
        {
            Finish();
            return;
        }

        _index = index;
        var pose = _poses[index];
        _phaseTime = 0.0;
        _fillHistory.Clear();
        _lastAskAt = double.NegativeInfinity;
        _ticks = 0;
        _saidOtherWay = false;
        _outcome = retry && _outcome != null && _outcome.Id == pose.Id ? _outcome : new PoseOutcome { Id = pose.Id };
        _measure = new PoseMeasure(pose.Id, _names, pose.IsRelax);
        Log((retry ? "retry " : _redoing ? "redo " : "pose ") + pose.Id);
        _message = pose.Prompt;
        Say(retry ? "Again. " + pose.Prompt : pose.Prompt);
        if (pose.IsRelax)
        {
            _ring = null;
            _phase = GuidedPhase.Relax;
            return;
        }

        var floor = _floor ?? Enumerable.Repeat(0.04, _n).ToArray();
        var channel = _info.Ceilings.Select(c => c * _scale).ToArray();
        _tref = (double[])_info.Ceilings.Clone();
        var typical = _data.Typical.GetValueOrDefault(pose.Id);
        var best = _history.Best.GetValueOrDefault(pose.Id);
        var index_ = _names.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        foreach (var target in pose.Targets)
        {
            var c = index_[target];
            if (best != null && best.TryGetValue(target, out var b) && b >= Math.Max(floor[c], 0.25 * _info.Ceilings[c]))
            {
                _tref[c] = b; // your best in this pose
            }
            else if (typical != null && typical.TryGetValue(target, out var ty))
            {
                _tref[c] = ty; // typical for this headset
            }
        }

        var tref = _tref.Select(v => v * _scale).ToArray();
        var companions = _data.Companions.GetValueOrDefault(pose.Id) ?? new HashSet<string>();
        _ring = new PoseRing(pose.Id, _names, pose.Halves, companions, channel, tref, floor, _jawRaw);
        _phase = GuidedPhase.Hold;
    }

    // a pose ends (done or skipped): keep its measurement and what it tells the next poses
    private void LeavePose()
    {
        var pose = _poses[_index];
        var ring = _ring!;
        ring.Finish();
        var outcome = _outcome!;
        outcome.Weak = ring.Halves.Any(h => h.Weak);
        foreach (var h in ring.Halves)
        {
            for (var k = 0; k < h.Columns.Length; k++)
            {
                if (!double.IsNegativeInfinity(h.Recorded[k]))
                {
                    outcome.Recorded[_names[h.Columns[k]]] = h.Recorded[k];
                }
            }

            if (!_redoing && h.Level is { } level)
            {
                _levels.Add(level * _scale);
            }
        }

        if (pose.Id == "jaw_open" && ring.Halves[0].Level is { } jaw && _tref != null)
        {
            _jawRaw = jaw * _scale * _tref[Array.IndexOf(_names, "JawOpen")];
        }

        if (_levels.Count >= FollowAfter)
        {
            _scale = Math.Min(Math.Max(Stats.Median(_levels), FollowMin), 1.0);
        }

        if (!outcome.Skipped && _measure!.Held(HzOf()) is { } held)
        {
            _run.Held[pose.Id] = held;
            if (pose.Id == "lips_pressed_open" && _measure.Ratio(HzOf()) is { } ratio)
            {
                _run.LipsRatio = ratio;
            }
        }

        if (_redoing)
        {
            outcome.Redone = true;
            _outcomes.RemoveAll(o => o.Id == outcome.Id);
        }

        _outcomes.Add(outcome);
    }

    // PoseMeasure's "enough frames" checks are per frame count; live frames come at the module's rate
    private double HzOf() => _measure!.Frames / Math.Max(_phaseTime, 1e-6);

    private void Finish(bool redo = false)
    {
        _run.RestP95 = Enumerable.Range(0, _n)
            .Select(i => _restFrames.Count > 0 ? Stats.Percentile(_restFrames.Select(r => r[i]).ToList(), 95) : 0.0)
            .ToArray();
        var poses = _poses.Where(p => !p.IsRelax).Select(p => (p.Id, p.Targets)).ToList();
        var runs = new List<RunMeasurement> { _run };
        runs.AddRange(_history.Runs);
        var now = _info.Ceilings;
        var (shapes, coupling) = PoseAnalysis.Analyze(_names, poses, runs, now, _info.Everyday);

        var c0 = (double[])_info.Priors.C0.Clone();
        var index = _names.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        foreach (var s in shapes)
        {
            c0[index[s.Name]] = s.Ceiling;
        }

        var coupled = _info.Priors.Coupled.ToDictionary(kv => kv.Key, kv => kv.Value);
        if (coupling is { } cr)
        {
            coupled["MouthClosed"] = ("JawOpen", cr);
        }

        var newPriors = new CalibrationPriors
        {
            Name = "personal-" + _info.ModuleKey,
            Expressions = _names,
            C0 = c0,
            B0 = _info.Priors.B0,
            Sigma0 = _info.Priors.Sigma0,
            Coupled = coupled,
            // normal use may always calm an expression back to where it was before this run (and earlier runs)
            CMax = now.Select((v, i) => Math.Max(v, _info.Priors.CMax?[i] ?? 0.0)).ToArray(),
            Source = $"guided calibration {_started:yyyy-MM-dd HH:mm}, {runs.Count} run(s) combined, from {_info.Priors.Name}",
        };

        _result = new GuidedResult
        {
            Started = _started,
            ModuleKey = _info.ModuleKey,
            Names = _names,
            Run = _run,
            Poses = _outcomes,
            Shapes = shapes,
            Coupling = coupling,
            Before = (double[])now.Clone(),
            NewPriors = newPriors,
            Summary = BuildSummary(shapes, index),
            RunsCombined = runs.Count,
        };
        _phase = GuidedPhase.Summary;
        Log(redo ? "summary after redo" : "summary");
        _message = "All done.";
        _ring = null;
        Say(redo ? "Updated. Have a look at the summary." : "All done. Have a look at the summary before you keep it.");
    }

    private List<SummaryRow> BuildSummary(List<ShapeResult> shapes, Dictionary<string, int> index)
    {
        var byName = shapes.ToDictionary(s => s.Name);
        var rows = new List<SummaryRow>();
        foreach (var pose in _poses.Where(p => !p.IsRelax))
        {
            var outcome = _outcomes.LastOrDefault(o => o.Id == pose.Id);
            var mine = pose.Targets.Where(byName.ContainsKey).Select(t => byName[t]).ToList();
            var ratios = mine.Select(s => s.Ceiling / Math.Max(_info.Ceilings[index[s.Name]], 1e-6)).ToList();
            var ratio = ratios.Count > 0 ? Math.Exp(ratios.Average(Math.Log)) : 1.0;
            var change = ratio > 1.05 ? "calmer" : ratio < 0.95 ? "stronger" : "about the same";
            if (outcome == null || outcome.Skipped)
            {
                rows.Add(new SummaryRow(pose.Id, pose.Title, change, "skipped, unchanged", SummaryColor.Grey));
                continue;
            }

            if (mine.Count > 0 && mine.All(s => s.Basis.StartsWith("not seen")))
            {
                rows.Add(new SummaryRow(pose.Id, pose.Title, change, "not sensed, unchanged", SummaryColor.Grey));
                continue;
            }

            string? why = null;
            if (mine.Any(s => s.Spread is > 1.15))
            {
                why = "differs between runs";
            }
            else if (outcome.Weak)
            {
                why = "weaker than your best";
            }
            else if (pose.Halves.Count == 2 && !pose.Sideways)
            {
                double Side(string side) => Math.Exp(pose.Halves[side].Where(byName.ContainsKey)
                    .Select(n => Math.Log(byName[n].Ceiling)).DefaultIfEmpty(0).Average());
                if (Math.Abs(Math.Log(Side("L") / Side("R"))) > Math.Log(1.15))
                {
                    why = "left and right differ";
                }
            }

            var redone = outcome.Redone ? ", redone" : "";
            rows.Add(why == null
                ? new SummaryRow(pose.Id, pose.Title, change, "measured" + redone, SummaryColor.Green)
                : new SummaryRow(pose.Id, pose.Title, change, why + ", worth a redo" + redone, SummaryColor.Yellow));
        }

        return rows;
    }

    private void Publish()
    {
        Volatile.Write(ref _snapshot, MakeSnapshot());
        if (_subscribed)
        {
            GuidedTap.Measuring = IsMeasuring();
        }
    }

    // Adaptive Calibration doesn't learn from the poses (they'd count as everyday expressions, and a discarded run
    // would still change the values in use). Not at the summary, while paused, or once a question has gone unanswered
    // for a minute: a run never skips by itself, so someone who walked away from it is back to normal use.
    private bool IsMeasuring() =>
        !_pausedByUser &&
        (_phase is GuidedPhase.Intro or GuidedPhase.Relax or GuidedPhase.Hold or GuidedPhase.Got ||
         _phase == GuidedPhase.Ask && _phaseTime - _lastAskAt < LeftAloneS);

    private GuidedSnapshot MakeSnapshot() => new(
        _phase,
        _index,
        _poses.Count,
        _index >= 0 && _index < _poses.Count ? _poses[_index] : null,
        _ring?.Halves.Select(h => new HalfView(h.Name, Math.Min(h.Fill, 1.0), h.Done, h.Counting, h.CurrentLevel)).ToList()
            ?? (IReadOnlyList<HalfView>)Array.Empty<HalfView>(),
        _trackingOk ? _message : "Tracking lost. Waiting for it to come back.",
        _alsoMoving,
        _trackingOk,
        _pausedByUser,
        _redoing,
        _result);
}
