using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using VRCFaceTracking.Core.Adaptive.Guided;

namespace VRCFaceTracking.Views.Guided;

/// <summary>
/// Draws a pose picture (<see cref="PosePictures"/>) and, around it, the Face ID-style ring: 48 ticks that light up as
/// the ring fills. Two-sided poses split it: the left half (the person's left, drawn like a mirror) fills from the top
/// going left, the right half from the top going right.
/// </summary>
public sealed class PoseCanvas : Control
{
    public static readonly StyledProperty<string?> PoseIdProperty =
        AvaloniaProperty.Register<PoseCanvas, string?>(nameof(PoseId));

    public static readonly StyledProperty<IReadOnlyList<HalfView>?> HalvesProperty =
        AvaloniaProperty.Register<PoseCanvas, IReadOnlyList<HalfView>?>(nameof(Halves));

    public static readonly StyledProperty<bool> ShowRingProperty =
        AvaloniaProperty.Register<PoseCanvas, bool>(nameof(ShowRing), true);

    public static readonly StyledProperty<bool> MistakeProperty =
        AvaloniaProperty.Register<PoseCanvas, bool>(nameof(Mistake));

    private const int Ticks = 48;
    private const double Box = 260.0; // drawing units: ring centre (130, 130), picture 120 x 120 scaled up in the middle

    private readonly Dictionary<string, Geometry> _geometry = new();

    static PoseCanvas()
    {
        AffectsRender<PoseCanvas>(PoseIdProperty, HalvesProperty, ShowRingProperty, MistakeProperty);
    }

    public string? PoseId
    {
        get => GetValue(PoseIdProperty);
        set => SetValue(PoseIdProperty, value);
    }

    public IReadOnlyList<HalfView>? Halves
    {
        get => GetValue(HalvesProperty);
        set => SetValue(HalvesProperty, value);
    }

    public bool ShowRing
    {
        get => GetValue(ShowRingProperty);
        set => SetValue(ShowRingProperty, value);
    }

    /// <summary> Draw the pose's "not this" picture instead. </summary>
    public bool Mistake
    {
        get => GetValue(MistakeProperty);
        set => SetValue(MistakeProperty, value);
    }

    private IBrush Brush(string key, Color fallback) =>
        this.TryFindResource(key, ActualThemeVariant, out var v) && v is IBrush b ? b : new SolidColorBrush(fallback);

    public override void Render(DrawingContext context)
    {
        var size = Math.Min(Bounds.Width, Bounds.Height);
        if (size <= 0 || PoseId == null)
        {
            return;
        }

        var scale = size / Box;
        var offset = new Point((Bounds.Width - size) / 2, (Bounds.Height - size) / 2);
        using var _ = context.PushTransform(Matrix.CreateScale(scale, scale) * Matrix.CreateTranslation(offset.X, offset.Y));

        var primary = Brush("TextFillColorPrimaryBrush", Colors.White);
        var secondary = Brush("TextFillColorSecondaryBrush", Colors.Gray);
        var accent = Brush("AccentFillColorDefaultBrush", Colors.DodgerBlue);
        var success = Brush("SystemFillColorSuccessBrush", Colors.SeaGreen);
        var idle = Brush("ControlStrongStrokeColorDefaultBrush", Colors.DimGray);

        if (ShowRing)
        {
            DrawRing(context, accent, success, idle);
        }

        var picture = Mistake ? PosePictures.MistakeFor(PoseId) : PosePictures.For(PoseId);
        if (picture == null)
        {
            return;
        }

        // the 120 x 120 picture, scaled to fit inside the ring
        using var __ = context.PushTransform(Matrix.CreateScale(1.6, 1.6) * Matrix.CreateTranslation(34, 34));
        foreach (var (ink, data) in picture)
        {
            var geometry = Geo(data);
            switch (ink)
            {
                case PictureInk.Fill:
                    context.DrawGeometry(primary, null, geometry);
                    break;
                default:
                    var (brush, width, dash) = ink switch
                    {
                        PictureInk.Face => (secondary, 1.5, (IDashStyle?)null),
                        PictureInk.Line => (primary, 2.5, null),
                        PictureInk.Thin => (primary, 1.2, null),
                        PictureInk.Arrow => (accent, 2.5, null),
                        _ => (secondary, 1.5, new DashStyle(new double[] { 3, 3 }, 0)),
                    };
                    context.DrawGeometry(null, new Pen(brush, width, dash, PenLineCap.Round, PenLineJoin.Round), geometry);
                    break;
            }
        }
    }

    private void DrawRing(DrawingContext context, IBrush filling, IBrush done, IBrush idle)
    {
        var halves = Halves ?? Array.Empty<HalfView>();
        var split = halves.Count == 2;
        for (var i = 0; i < Ticks; i++)
        {
            var a = (-90.0 + i * 360.0 / Ticks) * Math.PI / 180.0;
            HalfView? half;
            int position; // 0-based position of this tick within its half (or the whole ring), from the top
            int count;
            if (!split)
            {
                half = halves.Count > 0 ? halves[0] : null;
                position = i;
                count = Ticks;
            }
            else if (i < Ticks / 2)
            {
                // drawn like a mirror: the person's right side is on the right of the screen
                half = halves.FirstOrDefault(h => h.Name == "R");
                position = i;
                count = Ticks / 2;
            }
            else
            {
                half = halves.FirstOrDefault(h => h.Name == "L");
                position = Ticks - 1 - i;
                count = Ticks / 2;
            }

            var lit = half != null && position < Math.Round(half.Fill * count);
            var brush = !lit ? idle : half!.Done ? done : filling;
            var p1 = new Point(130 + 112 * Math.Cos(a), 130 + 112 * Math.Sin(a));
            var p2 = new Point(130 + 124 * Math.Cos(a), 130 + 124 * Math.Sin(a));
            context.DrawLine(new Pen(brush, 3.0, lineCap: PenLineCap.Round), p1, p2);
        }
    }

    private Geometry Geo(string data)
    {
        if (!_geometry.TryGetValue(data, out var g))
        {
            g = Geometry.Parse(data);
            _geometry[data] = g;
        }

        return g;
    }
}
