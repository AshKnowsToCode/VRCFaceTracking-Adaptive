namespace VRCFaceTracking.Views.Guided;

/// <summary> How a stroke of a pose picture is drawn. </summary>
public enum PictureInk
{
    Face, // outline, secondary colour
    Line, // feature, primary colour
    Thin, // fine detail
    Fill, // filled feature (eyes, pupils)
    Arrow, // the movement, accent colour
    Dash, // where something was / goes, dashed
}

/// <summary>
/// Plain line pictures of each guided pose (DESIGN.md 8.1): a face outline, the features that matter, and an arrow for
/// the movement, in a 120 x 120 box, written by hand as SVG-style path data (no image generator; nov-ash wants basic
/// drawings). Front views unless the movement is front-to-back. Some poses also have a "not this" picture of the
/// common mistake.
/// </summary>
public static class PosePictures
{
    private const string FaceOutline = "M22,60 A38,46 0 1 1 98,60 A38,46 0 1 1 22,60 Z";
    private const string Nose = "M60,56 L57,68 L62,68";
    private const string EyeLeft = "M43,50 A3,3 0 1 1 49,50 A3,3 0 1 1 43,50 Z";
    private const string EyeRight = "M71,50 A3,3 0 1 1 77,50 A3,3 0 1 1 71,50 Z";
    private const string Mouth = "M48,87 Q60,90 72,87";

    private static readonly (PictureInk, string)[] Front =
    [
        (PictureInk.Face, FaceOutline), (PictureInk.Fill, EyeLeft), (PictureInk.Fill, EyeRight), (PictureInk.Face, Nose),
    ];

    private static readonly (PictureInk, string)[] FrontNoEyes = [(PictureInk.Face, FaceOutline), (PictureInk.Face, Nose)];

    private static readonly (PictureInk, string)[] Profile =
    [
        (PictureInk.Face, "M34,14 C58,4 86,12 90,38 L91,46 L99,60 L92,64 L92,70 L90,72"),
        (PictureInk.Face, "M34,14 C14,26 12,58 26,78 L30,116"),
        (PictureInk.Face, "M39,58 A5,8 0 1 1 49,58 A5,8 0 1 1 39,58 Z"),
        (PictureInk.Line, "M78,42 L84,42"),
    ];

    private static (PictureInk, string)[] With((PictureInk, string)[] baseLayer, params (PictureInk, string)[] more) =>
        baseLayer.Concat(more).ToArray();

    private static readonly (PictureInk, string)[] Kiss = With(Front,
        (PictureInk.Line, "M54,86 A6,5 0 1 1 66,86 A6,5 0 1 1 54,86 Z"),
        (PictureInk.Thin, "M60,78 L60,75 M60,94 L60,97 M66,81 L68,79 M54,81 L52,79 M66,91 L68,93 M54,91 L52,93"),
        (PictureInk.Arrow, "M38,86 L48,86 M44,82 L48,86 L44,90 M82,86 L72,86 M76,82 L72,86 L76,90"));

    private static readonly (PictureInk, string)[] LipsRolledIn = With(Front,
        (PictureInk.Thin, "M46,86 L74,86"),
        (PictureInk.Arrow, "M50,76 Q58,76 58,83 M55,80 L58,83 L61,80 M70,96 Q62,96 62,89 M59,92 L62,89 L65,92"));

    private static readonly (PictureInk, string)[] TeethShowing = With(Front,
        (PictureInk.Line, "M42,79 L78,79 L78,93 L42,93 Z"),
        (PictureInk.Thin, "M42,86 L78,86 M49,79 L49,93 M55,79 L55,93 M61,79 L61,93 M67,79 L67,93 M73,79 L73,93"));

    private static readonly (PictureInk, string)[] Pout = With(Profile,
        (PictureInk.Line, "M90,72 Q101,71 97,80 L91,83 C91,91 84,96 75,97 L52,93 C46,100 44,108 44,116"),
        (PictureInk.Arrow, "M104,86 L104,76 M100,80 L104,76 L108,80"));

    private static readonly (PictureInk, string)[] Relaxed = With(Front, (PictureInk.Line, Mouth));

    private static readonly Dictionary<string, (PictureInk, string)[]> Pictures = new()
    {
        ["relax"] = Relaxed,
        ["relax_end"] = Relaxed,
        ["jaw_open"] = With(Front,
            (PictureInk.Line, "M51,90 A9,13 0 1 1 69,90 A9,13 0 1 1 51,90 Z"),
            (PictureInk.Arrow, "M60,108 L60,118 M56,114 L60,118 L64,114")),
        ["lips_pressed_open"] = With(Front,
            (PictureInk.Dash, "M30,96 Q60,124 90,96"),
            (PictureInk.Line, "M46,90 Q60,88 74,90"), (PictureInk.Line, "M46,92 Q60,94 74,92"),
            (PictureInk.Arrow, "M60,78 L60,86 M56,82 L60,86 L64,82 M60,104 L60,96 M56,100 L60,96 L64,100"),
            (PictureInk.Arrow, "M96,98 L96,110 M92,106 L96,110 L100,106")),
        ["funnel"] = With(Front,
            (PictureInk.Line, "M47,86 A13,11 0 1 1 73,86 A13,11 0 1 1 47,86 Z"),
            (PictureInk.Fill, "M55,86 A5,4 0 1 1 65,86 A5,4 0 1 1 55,86 Z"),
            (PictureInk.Arrow, "M71,77 L78,71 M73,70 L78,71 L77,76 M49,77 L42,71 M47,70 L42,71 L43,76"),
            (PictureInk.Arrow, "M71,95 L78,101 M73,102 L78,101 L77,96 M49,95 L42,101 M47,102 L42,101 L43,96")),
        ["pucker"] = Kiss,
        ["lips_in"] = LipsRolledIn,
        ["cheek_puff"] = With(Front,
            (PictureInk.Line, "M24,74 A12,14 0 1 1 48,74 A12,14 0 1 1 24,74 Z"),
            (PictureInk.Line, "M72,74 A12,14 0 1 1 96,74 A12,14 0 1 1 72,74 Z"),
            (PictureInk.Line, "M53,88 Q60,90 67,88"),
            (PictureInk.Arrow, "M22,74 L13,74 M17,70 L13,74 L17,78 M98,74 L107,74 M103,70 L107,74 L103,78")),
        ["cheek_suck"] = With(Front,
            (PictureInk.Line, "M34,64 Q44,75 36,88 M86,64 Q76,75 84,88"),
            (PictureInk.Line, "M54,88 A6,4 0 1 1 66,88 A6,4 0 1 1 54,88 Z"),
            (PictureInk.Arrow, "M20,76 L30,76 M26,72 L30,76 L26,80 M100,76 L90,76 M94,72 L90,76 L94,80")),
        ["smile"] = With(Front,
            (PictureInk.Line, "M40,82 Q60,104 80,82 Q60,90 40,82 Z"),
            (PictureInk.Thin, "M44,85 Q60,93 76,85"),
            (PictureInk.Arrow, "M37,80 L30,73 M30,79 L30,73 L36,73 M83,80 L90,73 M90,79 L90,73 L84,73")),
        ["upper_lip"] = With(Front,
            (PictureInk.Line, "M44,85 Q60,74 76,85"),
            (PictureInk.Thin, "M49,80 L71,80 L71,86 L49,86 Z"),
            (PictureInk.Line, "M44,89 Q60,93 76,89"),
            (PictureInk.Arrow, "M44,78 L44,70 M40,74 L44,70 L48,74 M76,78 L76,70 M72,74 L76,70 L80,74")),
        ["lower_lip"] = With(Front,
            (PictureInk.Line, "M44,84 Q60,86 76,84"),
            (PictureInk.Thin, "M49,86 L71,86 L71,91 L49,91 Z"),
            (PictureInk.Line, "M44,88 Q60,101 76,88"),
            (PictureInk.Arrow, "M60,100 L60,110 M56,106 L60,110 L64,106")),
        ["frown"] = With(Front,
            (PictureInk.Line, "M44,92 Q60,81 76,92"),
            (PictureInk.Arrow, "M42,94 L42,103 M38,99 L42,103 L46,99 M78,94 L78,103 M74,99 L78,103 L82,99")),
        ["stretch"] = With(Front,
            (PictureInk.Line, "M38,85 L82,85"), (PictureInk.Line, "M41,88 L79,88"),
            (PictureInk.Arrow, "M35,86 L25,86 M30,82 L25,86 L30,90 M85,86 L95,86 M90,82 L95,86 L90,90")),
        ["press"] = With(Front,
            (PictureInk.Line, "M44,85 Q60,83 76,85"), (PictureInk.Line, "M44,87 Q60,89 76,87"),
            (PictureInk.Arrow, "M60,71 L60,79 M56,75 L60,79 L64,75 M60,101 L60,93 M56,97 L60,93 L64,97")),
        ["chin_raise"] = Pout,
        ["sneer"] = With(Front,
            (PictureInk.Thin, "M54,60 Q60,57 66,60 M53,64 Q60,61 67,64"),
            (PictureInk.Line, "M46,87 Q52,80 60,83 Q68,80 74,87"),
            (PictureInk.Line, "M48,89 Q60,92 72,89"),
            (PictureInk.Arrow, "M48,68 L48,60 M44,64 L48,60 L52,64 M72,68 L72,60 M68,64 L72,60 L76,64")),
        ["jaw_sideways"] = With(Front,
            (PictureInk.Line, Mouth),
            (PictureInk.Dash, "M36,87 Q48,90 60,87 M60,87 Q72,90 84,87"),
            (PictureInk.Arrow, "M38,106 L82,106 M43,102 L38,106 L43,110 M77,102 L82,106 L77,110")),
        ["jaw_forward"] = With(Profile,
            (PictureInk.Dash, "M90,72 L92,75 L92,79 L90,82 C90,91 83,96 74,97 L52,93"),
            (PictureInk.Line, "M90,72 L98,74 L99,78 L97,82 C97,91 90,96 81,97 L56,93 C50,100 48,108 48,116"),
            (PictureInk.Arrow, "M90,106 L104,106 M99,102 L104,106 L99,110")),
        ["mouth_sideways"] = With(Front,
            (PictureInk.Line, Mouth),
            (PictureInk.Dash, "M36,87 Q48,90 60,87 M60,87 Q72,90 84,87"),
            (PictureInk.Arrow, "M30,96 L90,96 M35,92 L30,96 L35,100 M85,92 L90,96 L85,100")),
        ["squint"] = With(FrontNoEyes,
            (PictureInk.Line, "M40,50 Q46,46 52,50 Q46,52 40,50 Z M68,50 Q74,46 80,50 Q74,52 68,50 Z"),
            (PictureInk.Arrow, "M33,45 L39,50 L33,55 M87,45 L81,50 L87,55"),
            (PictureInk.Line, Mouth)),
        ["eyes_wide"] = With(FrontNoEyes,
            (PictureInk.Line, "M38,50 A8,8 0 1 1 54,50 A8,8 0 1 1 38,50 Z M66,50 A8,8 0 1 1 82,50 A8,8 0 1 1 66,50 Z"),
            (PictureInk.Fill, "M43,50 A3,3 0 1 1 49,50 A3,3 0 1 1 43,50 Z M71,50 A3,3 0 1 1 77,50 A3,3 0 1 1 71,50 Z"),
            (PictureInk.Arrow, "M46,38 L46,31 M43,34 L46,31 L49,34 M74,38 L74,31 M71,34 L74,31 L77,34"),
            (PictureInk.Line, Mouth)),
        ["brows_up"] = With(Front,
            (PictureInk.Line, "M38,36 Q46,30 54,36 M66,36 Q74,30 82,36"),
            (PictureInk.Arrow, "M46,27 L46,19 M42,23 L46,19 L50,23 M74,27 L74,19 M70,23 L74,19 L78,23"),
            (PictureInk.Line, Mouth)),
        ["brows_down"] = With(Front,
            (PictureInk.Line, "M38,38 L54,44 M82,38 L66,44"),
            (PictureInk.Arrow, "M42,29 L49,36 M49,31 L49,36 L44,36 M78,29 L71,36 M71,31 L71,36 L76,36"),
            (PictureInk.Line, Mouth)),
        ["tongue_out"] = With(Front,
            (PictureInk.Line, "M48,86 Q60,96 72,86 Q60,90 48,86 Z"),
            (PictureInk.Line, "M54,90 Q54,105 60,105 Q66,105 66,90"),
            (PictureInk.Arrow, "M60,108 L60,117 M56,113 L60,117 L64,113")),
    };

    private static readonly Dictionary<string, (PictureInk, string)[]> Mistakes = new()
    {
        ["press"] = LipsRolledIn,
        ["stretch"] = TeethShowing,
        ["funnel"] = Kiss,
        ["jaw_forward"] = Pout,
        ["cheek_puff"] = Kiss,
    };

    private static readonly Dictionary<string, string> MistakeCaptions = new()
    {
        ["press"] = "Not this: lips rolled in",
        ["stretch"] = "Not this: teeth showing",
        ["funnel"] = "Not this: a kiss",
        ["jaw_forward"] = "Not this: a pout",
        ["cheek_puff"] = "Not this: lips pushed forward",
    };

    public static IReadOnlyList<(PictureInk Ink, string Path)> For(string poseId) =>
        Pictures.TryGetValue(poseId, out var p) ? p : Relaxed;

    public static IReadOnlyList<(PictureInk Ink, string Path)>? MistakeFor(string poseId) =>
        Mistakes.GetValueOrDefault(poseId);

    public static string? MistakeCaption(string poseId) => MistakeCaptions.GetValueOrDefault(poseId);
}
