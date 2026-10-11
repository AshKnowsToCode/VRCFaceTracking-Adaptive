using System.Text;
using VRCFaceTracking.Core.Adaptive.Guided;
using VRCFaceTracking.Views.Guided;

/// <summary> AdaptiveReplay pictures OUT.html: every pose picture (and "not this" picture) on one page, for review. </summary>
internal static class PictureSheet
{
    public static int Run(string[] args)
    {
        var sb = new StringBuilder();
        sb.Append("<!doctype html><meta charset=utf-8><title>Pose pictures</title><style>")
          .Append("body{font-family:system-ui,sans-serif;background:#1f1f1f;color:#eee;margin:16px}")
          .Append(".g{display:flex;flex-wrap:wrap;gap:12px}.c{background:#2b2b2b;border-radius:8px;padding:8px;width:170px}")
          .Append(".c p{margin:4px 0 0;font-size:12px}.m{opacity:.7}</style><div class=g>");
        foreach (var pose in PoseLibrary.All)
        {
            sb.Append("<div class=c>").Append(Svg(PosePictures.For(pose.Id)))
              .Append($"<p><b>{pose.Title}</b></p><p>{pose.Prompt}</p>");
            if (PosePictures.MistakeFor(pose.Id) is { } m)
            {
                sb.Append(Svg(m, 90)).Append($"<p class=m>{PosePictures.MistakeCaption(pose.Id)}</p>");
            }

            sb.Append("</div>");
        }

        sb.Append("</div>");
        File.WriteAllText(args[0], sb.ToString());
        return 0;
    }

    private static string Svg(IReadOnlyList<(PictureInk Ink, string Path)> picture, int size = 150)
    {
        var sb = new StringBuilder($"<svg viewBox='0 0 120 120' width={size} height={size}>");
        foreach (var (ink, path) in picture)
        {
            var style = ink switch
            {
                PictureInk.Fill => "fill:#eee;stroke:none",
                PictureInk.Face => "fill:none;stroke:#999;stroke-width:1.5",
                PictureInk.Line => "fill:none;stroke:#eee;stroke-width:2.5",
                PictureInk.Thin => "fill:none;stroke:#eee;stroke-width:1.2",
                PictureInk.Arrow => "fill:none;stroke:#4aa3ff;stroke-width:2.5",
                _ => "fill:none;stroke:#999;stroke-width:1.5;stroke-dasharray:3 3",
            };
            sb.Append($"<path d='{path}' style='{style};stroke-linecap:round;stroke-linejoin:round'/>");
        }

        return sb.Append("</svg>").ToString();
    }
}
