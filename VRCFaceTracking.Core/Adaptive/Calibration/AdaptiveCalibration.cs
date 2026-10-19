using System.Diagnostics;
using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using VRCFaceTracking.Core.Adaptive.Guided;
using VRCFaceTracking.Core.Adaptive.Recording;
using VRCFaceTracking.Core.Params.Data;
using VRCFaceTracking.Core.Params.Data.Mutation;
using VRCFaceTracking.Core.Params.Expressions;
using StockCalibration = VRCFaceTracking.Core.Params.Data.Mutation.Calibration;

namespace VRCFaceTracking.Core.Adaptive.Calibration;

/// <summary>
/// Adaptive calibration v0 as a mutation: replaces the stock <see cref="StockCalibration"/> (DESIGN.md sections 6 and 10).
/// Starts from measured priors for the tracking module, then learns from normal use, bounded: an expression can only
/// get calmer than its starting value (at most 1.6x), never hotter.
/// </summary>
/// <remarks>
/// Learned state lives in its own file (<c>%APPDATA%\VRCFaceTracking-Adaptive\adaptive-calibration\</c>), not in
/// LocalSettings.json: written atomically every 60 active seconds and on exit, with one snapshot per day kept 14
/// days. It does nothing while the stock Calibration is on (both would scale the same values).
/// </remarks>
public class AdaptiveCalibration : TrackingMutation, IMutationTeardownHandler
{
    [MutationProperty("Pause learning (keep the current values)", true)]
    public bool pauseLearning = false;

    public override string Name => "Adaptive Calibration";

    public override string Description =>
        "Starts from measured values for your headset and learns from normal use. It can only make an expression " +
        "calmer than its start, never stronger. Replaces Calibration; switch that one off.";

    public override MutationPriority Step => MutationPriority.Preprocessor;

    // After the Session Recorder (int.MinValue) and before the stock Calibration (0): the guided calibration needs the
    // raw module values even while the stock Calibration is on (this one then passes data through to it).
    public override int Order => -1;

    public override bool IsSaved => true;

    public static readonly string StateDirectory = Path.Combine(Utils.PersistentDataDirectory, "adaptive-calibration");

    /// <summary> The live mutation, for the guided calibration page. </summary>
    public static AdaptiveCalibration? Current { get; private set; }

    private const double SaveEveryActiveS = 60.0;
    private const int SnapshotDays = 14;

    // Careful with new fields: MutationComponentFactory walks EVERY field of a mutation (private and static too) and
    // recurses into the objects they hold when the UI is built at load. Keep runtime objects null until the first
    // frame and never hold framework objects (JsonSerializerOptions crashed it) or other mutations in a field then.
    private readonly object _lock = new();
    private readonly object _writeLock = new();   // the periodic save on a worker vs a blocking one
    private AdaptiveCalibrator? _cal;
    private string? _moduleKey;
    private bool _modulesChanged = true;
    private double _lastSaveActiveS;
    private readonly float[] _in = new float[(int)UnifiedExpressions.Max];
    private readonly float[] _out = new float[(int)UnifiedExpressions.Max];
    private readonly long _clockStart = Stopwatch.GetTimestamp();
    private StockCalibration? _stock;
    private bool _stockWarned;
    private bool _faulted;
    private int _saving;
    private bool _exitHooked;
    // ceiling, guard factor and deadzone last written to the log, per expression (null until the first frame)
    private double[]? _loggedC;
    private double[]? _loggedG;
    private double[]? _loggedD;
    private int _loggedOcclMoments;
    private double _loggedOcclSeconds;

    public AdaptiveCalibration()
    {
        _isActive = false;
    }

    private static string[] ExpressionNames() =>
        Enumerable.Range(0, (int)UnifiedExpressions.Max).Select(i => ((UnifiedExpressions)i).ToString()).ToArray();

    public override void Initialize(UnifiedTrackingData data)
    {
        if (_exitHooked)
        {
            return;
        }

        _exitHooked = true;
        Current = this;
        AppDomain.CurrentDomain.ProcessExit += (_, _) => SaveNow(blocking: true);

        var modules = UnifiedTracking.Mutator?.LibManager?.LoadedModulesMetadata;
        if (modules != null)
        {
            modules.CollectionChanged += (_, _) => _modulesChanged = true;
        }
    }

    // First thing in teardown: a Windows shutdown ends the app before ProcessExit (2026-10-12, ~38 active s lost).
    public void OnTeardown()
    {
        if (SaveNow(blocking: true))
        {
            Logger?.LogInformation("Adaptive Calibration: state saved for shutdown.");
        }
    }

    public override void MutateData(ref UnifiedTrackingData data)
    {
        if (_faulted)
        {
            return;
        }

        try
        {
            var n = _in.Length;
            for (var i = 0; i < n; i++)
            {
                _in[i] = data.Shapes[i].Weight;
            }

            var t = (Stopwatch.GetTimestamp() - _clockStart) / (double)Stopwatch.Frequency;
            if (GuidedTap.HasListeners)
            {
                GuidedTap.Publish(t, _in); // raw values for a running guided calibration
            }

            if (StockCalibrationActive())
            {
                return;
            }

            lock (_lock)
            {
                if (_modulesChanged || _cal == null)
                {
                    SelectPriors();
                }

                var cal = _cal!;
                // a guided calibration's poses are calibrated as usual but never learned from (GuidedTap.Measuring)
                var guided = GuidedTap.Measuring;
                var learning = !pauseLearning && !guided;
                if (learning != cal.LearningEnabled)
                {
                    var what = learning ? "learning on" : guided ? "learning paused for the guided calibration" : "learning paused";
                    SessionRecorder.LogEvent("adaptive", what);
                    Logger?.LogInformation("Adaptive Calibration: {What}.", what);
                }

                cal.LearningEnabled = learning;
                cal.Step(t, _in, _out);
                for (var i = 0; i < n; i++)
                {
                    data.Shapes[i].Weight = _out[i];
                }

                if (cal.ActiveSeconds - _lastSaveActiveS >= SaveEveryActiveS)
                {
                    _lastSaveActiveS = cal.ActiveSeconds;
                    SaveNow(blocking: false);
                    LogChanges();
                }
            }
        }
        catch (Exception e)
        {
            // A calibration bug must never break tracking: stop calibrating, pass data through.
            _faulted = true;
            Logger?.LogError(e, "Adaptive Calibration failed and is passing data through unchanged until VRCFaceTracking restarts.");
            SessionRecorder.LogEvent("adaptive", "faulted");
        }
    }

    // Both would scale the same values, so this one steps aside while the stock Calibration is on.
    private bool StockCalibrationActive()
    {
        _stock ??= UnifiedTracking.Mutator?._mutations.OfType<StockCalibration>().FirstOrDefault();
        var on = _stock?.IsActive ?? false;
        if (on && !_stockWarned)
        {
            _stockWarned = true;
            Logger?.LogWarning("Adaptive Calibration is paused because Calibration is also on. Switch Calibration off to use Adaptive Calibration.");
        }
        else if (!on && _stockWarned)
        {
            _stockWarned = false;
            Logger?.LogInformation("Calibration is off; Adaptive Calibration is running.");
        }

        return on;
    }

    // Caller holds _lock. Picks the priors for the expression module and loads its saved state.
    private void SelectPriors()
    {
        _modulesChanged = false;
        var names = ExpressionNames();
        var modules = UnifiedTracking.Mutator?.LibManager?.LoadedModulesMetadata.ToArray() ?? [];
        CalibrationPriors? priors = null;
        string? key = null;
        foreach (var m in modules.Where(m => m.Active && m.UsingExpression))
        {
            priors = CalibrationPriors.ForModule(m.Name);
            if (priors != null)
            {
                key = m.Name;
                break;
            }
        }

        priors ??= CalibrationPriors.Identity(names);
        key ??= "identity";

        // a kept guided calibration replaces the module's starting values for this person
        var personal = PersonalPriorsPath(key);
        if (File.Exists(personal))
        {
            try
            {
                priors = CalibrationPriors.Load(personal);
            }
            catch (Exception e)
            {
                Logger?.LogWarning(e, "Adaptive Calibration couldn't read {Path}; using the module's starting values.", personal);
            }
        }
        if (_cal != null && key == _moduleKey)
        {
            return;
        }

        if (_cal != null)
        {
            SaveNow(blocking: true);
        }

        _moduleKey = key;
        _cal = new AdaptiveCalibrator(priors.Reorder(names));
        var path = StatePath(_cal.Priors.Name);
        var loaded = false;
        try
        {
            if (File.Exists(path))
            {
                var s = System.Text.Json.JsonSerializer.Deserialize<CalibrationState>(File.ReadAllText(path));
                loaded = s != null && _cal.LoadState(s);
            }
        }
        catch (Exception e)
        {
            Logger?.LogWarning(e, "Adaptive Calibration couldn't read {Path}; starting from the priors.", path);
        }

        _lastSaveActiveS = _cal.ActiveSeconds;
        RememberLogged();
        Logger?.LogInformation(
            "Adaptive Calibration: priors {Priors} for {Module}; {State}.", _cal.Priors.Name, key,
            loaded ? $"learned state loaded ({_cal.ActiveSeconds / 3600.0:0.0} active h)" : "no learned state yet");
        SessionRecorder.LogEvent("adaptive", $"priors {_cal.Priors.Name}; " + (loaded
            ? string.Create(CultureInfo.InvariantCulture, $"learned {_cal.ActiveSeconds:0.0} active s")
            : "no learned state"));
    }

    // Caller holds _lock. The values in use now become the baseline for LogChanges (no events).
    private void RememberLogged()
    {
        var cal = _cal!;
        var n = cal.Count;
        _loggedC = Enumerable.Range(0, n).Select(cal.Ceiling).ToArray();
        _loggedG = Enumerable.Range(0, n).Select(cal.Guard).ToArray();
        _loggedD = Enumerable.Range(0, n).Select(i => cal.Params.DeadzoneK * cal.Noise(i)).ToArray();
        _loggedOcclMoments = cal.OcclusionMoments;
        _loggedOcclSeconds = cal.OcclusionPausedSeconds;
    }

    // Caller holds _lock. Every 60 active s: what moved since it was last logged goes to the recording (adaptive:ceiling,
    // adaptive:guard, adaptive:deadzone) and one app-log line. Ceiling or guard factor by >= 4% (one guardrail step is
    // 5%), deadzone by >= 0.01. Without this, drift and the 2026-10-13 deadzone growth were only visible in a replay.
    private void LogChanges()
    {
        var cal = _cal!;
        if (_loggedC == null || _loggedG == null || _loggedD == null)
        {
            RememberLogged();
            return;
        }

        StringBuilder? summary = null;
        void Note(string what, int i, double old, double now)
        {
            var text = string.Create(CultureInfo.InvariantCulture,
                $"{what} {cal.Priors.Expressions[i]} {old:0.000} -> {now:0.000}");
            SessionRecorder.LogEvent("adaptive", text);
            summary = summary == null ? new StringBuilder(text) : summary.Append("; ").Append(text);
        }

        for (var i = 0; i < cal.Count; i++)
        {
            var c = cal.Ceiling(i);
            var g = cal.Guard(i);
            var d = cal.Params.DeadzoneK * cal.Noise(i);
            if (Math.Abs(c / _loggedC[i] - 1.0) >= 0.04)
            {
                Note("ceiling", i, _loggedC[i], c);
                _loggedC[i] = c;
            }

            if (Math.Abs(g / _loggedG[i] - 1.0) >= 0.04)
            {
                Note("guard", i, _loggedG[i], g);
                _loggedG[i] = g;
            }

            if (Math.Abs(d - _loggedD[i]) >= 0.01)
            {
                Note("deadzone", i, _loggedD[i], d);
                _loggedD[i] = d;
            }
        }

        // occlusion gate (cup, hand or food at the mouth): how often it kept data out of learning since the last line
        var moments = cal.OcclusionMoments - _loggedOcclMoments;
        if (moments > 0)
        {
            var text = string.Create(CultureInfo.InvariantCulture,
                $"occlusion {moments} moments, {cal.OcclusionPausedSeconds - _loggedOcclSeconds:0.0} s learning paused");
            SessionRecorder.LogEvent("adaptive", text);
            summary = summary == null ? new StringBuilder(text) : summary.Append("; ").Append(text);
            _loggedOcclMoments = cal.OcclusionMoments;
            _loggedOcclSeconds = cal.OcclusionPausedSeconds;
        }

        if (summary != null)
        {
            Logger?.LogInformation("Adaptive Calibration learned: {Changes}", summary.ToString());
        }
    }

    public static string PersonalPriorsPath(string moduleKey) =>
        Path.Combine(StateDirectory, $"personal-{Sanitize(moduleKey)}.json");

    /// <summary> What a guided calibration needs from the live calibrator, or null before tracking data arrived. </summary>
    public GuidedInfo? GetGuidedInfo()
    {
        lock (_lock)
        {
            if (_cal == null || _moduleKey == null)
            {
                return null;
            }

            var n = _cal.Count;
            return new GuidedInfo(
                _moduleKey,
                _cal.Priors,
                Enumerable.Range(0, n).Select(i => _cal.Ceiling(i) * _cal.Guard(i)).ToArray(),
                Enumerable.Range(0, n).Select(_cal.MaxHeldPeak).ToArray(),
                _cal.ActiveSeconds);
        }
    }

    /// <summary>
    /// Makes a kept guided calibration the starting values for this person: written to the personal priors file,
    /// the learned state starts over from them (the old state stays in its file and snapshots).
    /// </summary>
    public void ApplyPersonalPriors(CalibrationPriors priors)
    {
        SaveNow(blocking: true);
        lock (_lock)
        {
            var key = _moduleKey ?? "identity";
            Directory.CreateDirectory(StateDirectory);
            var path = PersonalPriorsPath(key);
            var tmp = path + ".tmp";
            File.WriteAllText(tmp, priors.ToJson(), new UTF8Encoding(false));
            File.Move(tmp, path, overwrite: true);
            var old = _cal;
            _cal = new AdaptiveCalibrator(priors.Reorder(ExpressionNames()));
            if (old != null)
            {
                _cal.CarryOver(old); // keep the everyday knowledge (rest, noise, held peaks); ceilings start anew
            }

            var statePath = StatePath(_cal.Priors.Name);
            if (File.Exists(statePath))
            {
                var keep = Path.Combine(StateDirectory, "snapshots",
                    $"{Path.GetFileNameWithoutExtension(statePath)}-before-guided-{DateTime.Now:yyyyMMdd-HHmmss}.json");
                Directory.CreateDirectory(Path.GetDirectoryName(keep)!);
                File.Move(statePath, keep, overwrite: true);
            }

            _lastSaveActiveS = 0.0;
            RememberLogged();
            Logger?.LogInformation("Adaptive Calibration now starts from your guided calibration ({Priors}).", priors.Name);
            SessionRecorder.LogEvent("adaptive", "guided applied " + priors.Name);
        }

        SaveNow(blocking: true);
    }

    private static string StatePath(string priorsName) =>
        Path.Combine(StateDirectory, $"state-{Sanitize(priorsName)}.json");

    private static string Sanitize(string s) =>
        string.Concat(s.Select(ch => char.IsLetterOrDigit(ch) || ch is '-' or '_' or '.' ? ch : '_'));

    // Snapshot under the lock (cheap), write on a worker unless blocking (exit, priors switch, buttons). Blocking:
    // whether the state reached the disk.
    private bool SaveNow(bool blocking)
    {
        CalibrationState state;
        string path;
        lock (_lock)
        {
            if (_cal == null)
            {
                return false;
            }

            state = _cal.SaveState();
            state.SavedUtc = DateTime.UtcNow;
            path = StatePath(_cal.Priors.Name);
        }

        if (blocking)
        {
            return Write(state, path);
        }

        if (Interlocked.Exchange(ref _saving, 1) == 1)
        {
            return false; // the previous write is still running; the next interval catches up
        }

        Task.Run(() =>
        {
            try
            {
                Write(state, path);
            }
            finally
            {
                Volatile.Write(ref _saving, 0);
            }
        });
        return true;
    }

    private bool Write(CalibrationState state, string path)
    {
        lock (_writeLock)
        {
            return WriteLocked(state, path);
        }
    }

    private bool WriteLocked(CalibrationState state, string path)
    {
        try
        {
            Directory.CreateDirectory(StateDirectory);
            var json = System.Text.Json.JsonSerializer.Serialize(state);
            var tmp = path + ".tmp";
            File.WriteAllText(tmp, json, new UTF8Encoding(false));
            File.Move(tmp, path, overwrite: true);

            var snapDir = Path.Combine(StateDirectory, "snapshots");
            Directory.CreateDirectory(snapDir);
            var stem = Path.GetFileNameWithoutExtension(path);
            var snap = Path.Combine(snapDir, $"{stem}-{DateTime.Now:yyyyMMdd}.json");
            if (!File.Exists(snap))
            {
                File.Copy(path, snap);
                foreach (var old in Directory.GetFiles(snapDir, stem + "-*.json"))
                {
                    if (File.GetLastWriteTime(old) < DateTime.Now.AddDays(-SnapshotDays))
                    {
                        File.Delete(old);
                    }
                }
            }
            return true;
        }
        catch (Exception e)
        {
            Logger?.LogWarning(e, "Adaptive Calibration couldn't save its state to {Path}.", path);
            return false;
        }
    }

    [MutationButton("Back to starting values")]
    public void ResetToPriors()
    {
        lock (_lock)
        {
            if (_cal == null)
            {
                Logger?.LogInformation("Adaptive Calibration hasn't started yet; nothing to reset.");
                return;
            }

            var path = StatePath(_cal.Priors.Name);
            try
            {
                if (File.Exists(path))
                {
                    var keep = Path.Combine(StateDirectory, "snapshots",
                        $"{Path.GetFileNameWithoutExtension(path)}-before-reset-{DateTime.Now:yyyyMMdd-HHmmss}.json");
                    Directory.CreateDirectory(Path.GetDirectoryName(keep)!);
                    File.Copy(path, keep, overwrite: true);
                }
            }
            catch (Exception e)
            {
                Logger?.LogWarning(e, "Adaptive Calibration couldn't keep a copy of the old state.");
            }

            _cal = new AdaptiveCalibrator(_cal.Priors, _cal.Params);
            _lastSaveActiveS = 0.0;
            RememberLogged();
            Logger?.LogInformation("Adaptive Calibration is back to the starting values of {Priors} (old state kept in snapshots).", _cal.Priors.Name);
            SessionRecorder.LogEvent("adaptive", "reset " + _cal.Priors.Name);
        }

        SaveNow(blocking: true);
    }

    [MutationButton("Forget guided calibration")]
    public void ForgetPersonalPriors()
    {
        string? key;
        lock (_lock)
        {
            key = _moduleKey;
        }

        var path = key != null ? PersonalPriorsPath(key) : null;
        if (path == null || !File.Exists(path))
        {
            Logger?.LogInformation("Adaptive Calibration has no guided calibration to forget.");
            return;
        }

        SaveNow(blocking: true);
        var keep = Path.Combine(StateDirectory, "snapshots",
            $"{Path.GetFileNameWithoutExtension(path)}-forgotten-{DateTime.Now:yyyyMMdd-HHmmss}.json");
        Directory.CreateDirectory(Path.GetDirectoryName(keep)!);
        File.Move(path, keep, overwrite: true);
        lock (_lock)
        {
            _cal = null; // the next frame picks the module's starting values again
            _modulesChanged = true;
        }

        Logger?.LogInformation("Adaptive Calibration forgot the guided calibration (copy kept in snapshots); back to the module's starting values.");
        SessionRecorder.LogEvent("adaptive", "forgot guided " + Path.GetFileNameWithoutExtension(path));
    }

    [MutationButton("Show what it learned (log)")]
    public void LogStatus()
    {
        lock (_lock)
        {
            if (_cal == null)
            {
                Logger?.LogInformation("Adaptive Calibration hasn't received tracking data yet.");
                return;
            }

            var sb = new StringBuilder();
            sb.Append(CultureInfo.InvariantCulture,
                $"Adaptive Calibration: priors {_cal.Priors.Name}, {_cal.ActiveSeconds / 3600.0:0.00} active h, learning {(pauseLearning ? "paused" : "on")}.");
            var changed = Enumerable.Range(0, _cal.Count).Where(i => Math.Abs(_cal.CeilingRatio(i) - 1.0) > 0.01).ToArray();
            sb.Append(CultureInfo.InvariantCulture, $" Calmer than the start: {changed.Length} of {_cal.Count} expressions.");
            foreach (var i in changed.OrderByDescending(i => _cal.CeilingRatio(i)))
            {
                sb.Append(CultureInfo.InvariantCulture,
                    $"\n  {_cal.Priors.Expressions[i]}: {_cal.CeilingRatio(i):0.00}x calmer (ceiling {_cal.Ceiling(i):0.000}, guard {_cal.Guard(i):0.00})");
            }

            Logger?.LogInformation(sb.ToString());
        }
    }

    [MutationButton("Open calibration folder")]
    public void OpenStateFolder()
    {
        Directory.CreateDirectory(StateDirectory);
        Process.Start(new ProcessStartInfo { FileName = StateDirectory, UseShellExecute = true });
    }
}

/// <summary> The live calibrator's view for a guided calibration (see <see cref="AdaptiveCalibration.GetGuidedInfo"/>). </summary>
/// <param name="ModuleKey">Tracking module the priors were chosen for ("identity" when none).</param>
/// <param name="Priors">Starting values in use, in expression order.</param>
/// <param name="Ceilings">Ceilings in effect now (learned ceiling x guardrail), per expression.</param>
/// <param name="Everyday">Highest held peak seen in normal use, per expression (0 = none yet).</param>
/// <param name="ActiveSeconds">How much normal use the calibrator has learned from.</param>
public sealed record GuidedInfo(string ModuleKey, CalibrationPriors Priors, double[] Ceilings, double[] Everyday,
    double ActiveSeconds = 0);
