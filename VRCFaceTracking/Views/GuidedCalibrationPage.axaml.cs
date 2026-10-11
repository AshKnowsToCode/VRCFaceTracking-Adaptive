using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using VRCFaceTracking.Contracts;
using VRCFaceTracking.Core.Adaptive.Calibration;
using VRCFaceTracking.Core.Adaptive.Guided;
using VRCFaceTracking.Views.Guided;
using StockCalibration = VRCFaceTracking.Core.Params.Data.Mutation.Calibration;

namespace VRCFaceTracking.Views;

/// <summary>
/// Adaptive-calibration fork: the guided calibration (DESIGN.md 8.1, lean first release). A run measures each pose
/// with a Face ID-style ring, asks instead of skipping when a pose doesn't register, and ends on a summary; Keep makes
/// the results the Adaptive Calibration's starting values. Every finished run is saved as a report.
/// </summary>
public partial class GuidedCalibrationPage : UserControl, INotifyNavigated
{
    private readonly DispatcherTimer _timer;
    private const double EverydayMinutes = 30.0; // below this much learned normal use the page suggests waiting

    // Testing aid: VRCFT_GUIDED_DEMO=1 runs the page on made-up face data (no headset, no settings changes, silent).
    private static readonly bool Demo = Environment.GetEnvironmentVariable("VRCFT_GUIDED_DEMO") == "1";
    private DispatcherTimer? _demoTimer;
    private readonly Random _demoRandom = new(1);
    private double _demoT;
    private double _demoPoseStart;
    private string? _demoPose;
    private bool _demoStarted;
    private readonly GuidedVoice _voice = new();
    private GuidedSession? _session;
    private bool _summaryShown;
    private GuidedResult? _shownResult;
    private string? _shownPose;

    public GuidedCalibrationPage()
    {
        InitializeComponent();
        _timer = new DispatcherTimer(TimeSpan.FromMilliseconds(33), DispatcherPriority.Normal, (_, _) => Tick());
        VoiceCheck.IsCheckedChanged += (_, _) => _voice.Enabled = VoiceCheck.IsChecked == true;
        _timer.Start();
    }

    public void OnNavigatedTo()
    {
        if (_session == null)
        {
            RefreshHistory();
        }
    }

    private static StockCalibration? Stock() =>
        UnifiedTracking.Mutator?._mutations.OfType<StockCalibration>().FirstOrDefault();

    // ------------------------------------------------------------------------------------------------------ start

    private void RefreshStart()
    {
        var adaptive = AdaptiveCalibration.Current;
        var modules = UnifiedTracking.Mutator?.LibManager?.LoadedModulesMetadata.Where(m => m.Active).ToArray() ?? [];
        var eyes = modules.Where(m => m.UsingEye).Select(m => m.Name).Distinct().ToArray();
        var face = modules.Where(m => m.UsingExpression).Select(m => m.Name).Distinct().ToArray();
        TrackingLine.Text = (eyes, face) switch
        {
            ({ Length: 0 }, { Length: 0 }) => "No tracking module is running.",
            _ when eyes.SequenceEqual(face) => $"Eyes and face: {string.Join(", ", face)}",
            _ => string.Join(";  ", new[]
            {
                face.Length > 0 ? $"Face: {string.Join(", ", face)}" : "No face tracking",
                eyes.Length > 0 ? $"Eyes: {string.Join(", ", eyes)}" : "No eye tracking",
            }),
        };

        var stockOn = Stock()?.IsActive ?? false;
        string ready;
        var canStart = false;
        if (Demo && adaptive?.GetGuidedInfo() == null)
        {
            ready = "Demo mode: made-up face data.";
            canStart = true;
        }
        else if (adaptive == null)
        {
            ready = "Adaptive Calibration isn't loaded.";
        }
        else if (!adaptive.IsActive || stockOn)
        {
            ready = "The results become Adaptive Calibration's starting values, so it needs to be on (and the old " +
                    "Calibration off, so you see the new values).";
        }
        else if (Demo)
        {
            ready = "Demo mode: made-up face data.";
            canStart = true;
        }
        else if (adaptive.GetGuidedInfo() == null)
        {
            ready = "Waiting for tracking data. Put the headset on.";
        }
        else
        {
            var minutes = adaptive.GetGuidedInfo()!.ActiveSeconds / 60.0;
            ready = minutes < EverydayMinutes
                ? $"Ready. Tip: Adaptive Calibration has seen only {minutes:0} minutes of normal use. Smiles and other " +
                  "everyday expressions are measured from normal use, so the result is best after about half an hour " +
                  "of using the headset normally (talking, laughing) with Adaptive Calibration on. You can start now " +
                  "anyway; normal use will correct those expressions later."
                : "Ready.";
            canStart = true;
        }

        ReadyLine.Text = ready;
        UseAdaptiveButton.IsVisible = adaptive != null && (!adaptive.IsActive || stockOn);
        StartButton.IsEnabled = canStart;
    }

    private void RefreshHistory()
    {
        HistoryList.Children.Clear();
        var reports = GuidedStore.ListReports();
        if (reports.Count == 0)
        {
            HistoryList.Children.Add(new TextBlock { Text = "None yet.", Opacity = 0.6 });
            return;
        }

        foreach (var r in reports.Take(20))
        {
            HistoryList.Children.Add(new TextBlock
            {
                Text = $"{r.Started:yyyy-MM-dd HH:mm}   {(r.Kept ? "kept" : "discarded")}   " +
                       $"{r.Measured} measured, {r.Skipped} skipped, {r.Asked} questions   ({r.Module})",
                Opacity = r.Kept ? 0.9 : 0.6,
            });
        }
    }

    private void OnUseAdaptive(object? sender, RoutedEventArgs e)
    {
        if (AdaptiveCalibration.Current is { } adaptive)
        {
            adaptive.IsActive = true;
        }

        if (Stock() is { IsActive: true } stock)
        {
            stock.IsActive = false;
        }

        RefreshStart();
    }

    private void OnStart(object? sender, RoutedEventArgs e)
    {
        var info = Demo ? DemoInfo() : AdaptiveCalibration.Current?.GetGuidedInfo();
        if (info == null)
        {
            RefreshStart();
            return;
        }

        var history = GuidedStore.LoadHistory(info.ModuleKey, info.Priors.Expressions);
        _session = new GuidedSession(info, PoseLibrary.DefaultRun(info.ModuleKey), GuidedModuleData.ForModule(info.ModuleKey),
            history);
        _session.Cue += _voice.Play;
        _summaryShown = false;
        _shownPose = null;
        StartPanel.IsVisible = false;
        SummaryPanel.IsVisible = false;
        RunPanel.IsVisible = true;
        _session.Start();
        if (Demo)
        {
            VoiceCheck.IsChecked = false;
            _demoTimer = new DispatcherTimer(TimeSpan.FromMilliseconds(16), DispatcherPriority.Normal, (_, _) => DemoFrame());
            _demoTimer.Start();
        }
    }

    private static GuidedInfo DemoInfo()
    {
        var names = Enumerable.Range(0, (int)Core.Params.Expressions.UnifiedExpressions.Max)
            .Select(i => ((Core.Params.Expressions.UnifiedExpressions)i).ToString()).ToArray();
        var priors = (CalibrationPriors.ForModule("Virtual Desktop") ?? CalibrationPriors.Identity(names)).Reorder(names);
        return new GuidedInfo("Virtual Desktop", priors, priors.C0.Select(c => Math.Clamp(c, 0.15, 1.0)).ToArray(),
            new double[names.Length]);
    }

    // a face that does each pose about 0.6 s after it is asked (sideways: left first, then right)
    private void DemoFrame()
    {
        if (_session == null)
        {
            return;
        }

        _demoT += 0.016;
        var s = _session.Snapshot;
        var names = DemoInfo().Priors.Expressions;
        var x = new float[names.Length];
        for (var i = 0; i < x.Length; i++)
        {
            x[i] = (float)(0.01 + 0.01 * _demoRandom.NextDouble());
        }

        if (s.Pose is { IsRelax: false } pose && s.Phase is GuidedPhase.Hold or GuidedPhase.Ask)
        {
            if (pose.Id != _demoPose)
            {
                _demoPose = pose.Id;
                _demoPoseStart = _demoT;
            }

            var held = _demoT - _demoPoseStart;
            var ramp = Math.Clamp((held - 0.6) / 0.4, 0.0, 1.0);
            var typical = GuidedModuleData.ForModule("Virtual Desktop").Typical.GetValueOrDefault(pose.Id);
            var shapes = pose.Sideways ? pose.Halves[held < 3.5 ? "L" : "R"] : pose.Targets;
            foreach (var shape in shapes)
            {
                var i = Array.IndexOf(names, shape);
                var v = typical != null && typical.TryGetValue(shape, out var ty) ? ty : 0.8;
                x[i] = (float)(v * ramp * (1 + 0.02 * _demoRandom.NextDouble()));
            }

            if (pose.Id == "lips_pressed_open")
            {
                x[Array.IndexOf(names, "JawOpen")] = (float)(0.5 * ramp);
            }
        }

        _session.Feed(_demoT, x);
    }

    // -------------------------------------------------------------------------------------------------------- run

    private void Tick()
    {
        if (_session == null)
        {
            if (StartPanel.IsVisible)
            {
                RefreshStart();
                if (Demo && !_demoStarted && IsVisible)
                {
                    _demoStarted = true; // the demo starts by itself
                    OnStart(null, new RoutedEventArgs());
                }
            }

            return;
        }

        var s = _session.Snapshot;
        if (s.Phase == GuidedPhase.Cancelled)
        {
            EndSession();
            return;
        }

        if (s.Phase == GuidedPhase.Summary && s.Result != null)
        {
            if (!_summaryShown || !ReferenceEquals(s.Result, _shownResult))
            {
                ShowSummary(s.Result); // first time, or worked out again after a redo
            }

            return;
        }

        if (!RunPanel.IsVisible)
        {
            // a redo from the summary
            SummaryPanel.IsVisible = false;
            RunPanel.IsVisible = true;
            _shownPose = null;
        }

        var pose = s.Pose;
        if (pose != null && pose.Id != _shownPose)
        {
            _shownPose = pose.Id;
            Canvas.PoseId = pose.Id;
            PoseTitle.Text = pose.Title;
            PosePrompt.Text = pose.Prompt;
            PoseMistake.Text = pose.Mistake;
            var caption = PosePictures.MistakeCaption(pose.Id);
            MistakeBox.IsVisible = caption != null;
            MistakeCanvas.PoseId = pose.Id;
            MistakeCaption.Text = caption ?? "";
        }

        var poses = _session.Poses.Where(p => !p.IsRelax).ToList();
        Counter.Text = s.Redoing ? "Redo (optional): skip or stop keeps the earlier result"
            : s.Phase == GuidedPhase.Intro ? "Starting"
            : pose is { IsRelax: true } ? "Relax"
            : $"Pose {poses.IndexOf(pose!) + 1} of {poses.Count}";
        Overall.Value = s.PoseCount > 0 ? Math.Max(s.PoseIndex, 0) / (double)s.PoseCount : 0;
        Canvas.Halves = s.Halves;
        Canvas.ShowRing = s.Phase is GuidedPhase.Hold or GuidedPhase.Ask or GuidedPhase.Got;
        var level = s.Halves.Where(h => !h.Done).Select(h => h.Level).DefaultIfEmpty(1.0).Max();
        LevelBox.IsVisible = s.Halves.Count > 0;
        Level.Value = Math.Clamp(level, 0, 1.2);
        LevelText.Text = $"{Math.Round(level * 100)}% of your best";
        Message.Text = s.PausedByUser ? "Paused." : s.Message;
        AlsoMoving.Text = s.AlsoMoving != null ? $"Also moving: {s.AlsoMoving}" : "";
        PauseButton.Content = s.PausedByUser ? "Resume" : "Pause";
        RetryButton.IsVisible = s.Phase == GuidedPhase.Ask;
        SkipButton.IsEnabled = s.Phase is GuidedPhase.Hold or GuidedPhase.Ask;
    }

    private void OnPause(object? sender, RoutedEventArgs e) =>
        _session?.SetPaused(!(_session.Snapshot.PausedByUser));

    private void OnRetry(object? sender, RoutedEventArgs e) => _session?.Retry();

    private void OnSkip(object? sender, RoutedEventArgs e) => _session?.Skip();

    private void OnStop(object? sender, RoutedEventArgs e) => _session?.Cancel();

    // ---------------------------------------------------------------------------------------------------- summary

    private void ShowSummary(GuidedResult result)
    {
        _summaryShown = true;
        _shownResult = result;
        RunPanel.IsVisible = false;
        SummaryPanel.IsVisible = true;
        KeepButton.IsEnabled = true;
        DiscardButton.IsEnabled = true;
        KeptNote.Text = "";
        var measured = result.Summary.Count(r => r.Color != SummaryColor.Grey);
        SummaryTitle.Text = $"{measured} of {result.Summary.Count} measured";
        SummaryNote.Text = (result.RunsCombined > 1 ? $"Combined with your last {result.RunsCombined - 1} kept run(s). " : "") +
                           "Calmer = the avatar moves less for the same face; stronger = more. Nothing changes until you press Keep.";
        SummaryRows.Children.Clear();
        foreach (var row in result.Summary)
        {
            var color = row.Color switch
            {
                SummaryColor.Green => Color.FromRgb(0x3B, 0xA5, 0x5C),
                SummaryColor.Yellow => Color.FromRgb(0xE0, 0xA8, 0x00),
                _ => Color.FromRgb(0x80, 0x80, 0x80),
            };
            var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("Auto,*,Auto,Auto,Auto"), Margin = new Thickness(0, 2) };
            grid.Children.Add(new Ellipse { Width = 10, Height = 10, Fill = new SolidColorBrush(color), Margin = new Thickness(0, 0, 10, 0), VerticalAlignment = VerticalAlignment.Center });
            var title = new TextBlock { Text = row.Title, VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(title, 1);
            var change = new TextBlock { Text = row.Change, Opacity = 0.85, Margin = new Thickness(12, 0), VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(change, 2);
            var status = new TextBlock { Text = row.Status, Opacity = 0.6, VerticalAlignment = VerticalAlignment.Center };
            Grid.SetColumn(status, 3);
            var poseId = row.PoseId;
            var redo = new Button
            {
                Content = "Redo (optional)",
                FontSize = 12,
                Padding = new Thickness(8, 2),
                Margin = new Thickness(12, 0, 0, 0),
                VerticalAlignment = VerticalAlignment.Center,
            };
            ToolTip.SetTip(redo, "Do just this pose again. Skip or Stop keeps the earlier result.");
            redo.Click += (_, _) => _session?.Redo(poseId);
            Grid.SetColumn(redo, 4);
            grid.Children.Add(title);
            grid.Children.Add(change);
            grid.Children.Add(status);
            grid.Children.Add(redo);
            SummaryRows.Children.Add(grid);
        }
    }

    private void OnKeep(object? sender, RoutedEventArgs e)
    {
        var result = _session?.Snapshot.Result;
        if (result == null)
        {
            return;
        }

        KeepButton.IsEnabled = DiscardButton.IsEnabled = false;
        try
        {
            GuidedStore.SaveReport(result, kept: true);
            AdaptiveCalibration.Current?.ApplyPersonalPriors(result.NewPriors);
            KeptNote.Text = "Kept. Adaptive Calibration now starts from these values and keeps learning from normal use. " +
                            "To undo: Tracking settings > Adaptive Calibration > Forget guided calibration.";
        }
        catch (Exception ex)
        {
            KeptNote.Text = "Couldn't save: " + ex.Message;
            KeepButton.IsEnabled = DiscardButton.IsEnabled = true;
            return;
        }

        EndSession(keepSummary: true);
    }

    private void OnDiscard(object? sender, RoutedEventArgs e)
    {
        var result = _session?.Snapshot.Result;
        if (result != null)
        {
            GuidedStore.SaveReport(result, kept: false);
        }

        EndSession();
    }

    private void EndSession(bool keepSummary = false)
    {
        _demoTimer?.Stop();
        _demoTimer = null;
        if (_session != null)
        {
            _session.Cue -= _voice.Play;
            _session.Dispose();
            _session = null;
        }

        RunPanel.IsVisible = false;
        SummaryPanel.IsVisible = keepSummary;
        StartPanel.IsVisible = true;
        RefreshHistory();
        RefreshStart();
    }
}
