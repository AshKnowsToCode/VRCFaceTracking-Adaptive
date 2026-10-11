namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary> One guided pose: what to say, which expressions it measures, and how its ring is split. </summary>
public sealed record Pose(
    string Id,
    string Title,
    string Prompt,
    string Mistake,
    string[] Targets,
    IReadOnlyDictionary<string, string[]> Halves,
    bool Sideways)
{
    /// <summary> A relax step: no targets, measures the person's resting face (floors, noise). </summary>
    public bool IsRelax => Targets.Length == 0;
}

/// <summary>
/// Every pose the guided calibration knows (DESIGN.md 8.1: a library; each module's default list picks from it).
/// Ids and targets match tools/replay/poses.py POSES and ring_sim.build_steps (sideways moves are one step each).
/// </summary>
public static class PoseLibrary
{
    private static readonly string[] Sides = ["Left", "Right"];

    private static string[] Both(params string[] stems) => stems.SelectMany(s => Sides.Select(side => s + side)).ToArray();

    private static Pose Whole(string id, string title, string prompt, string mistake, params string[] targets) =>
        new(id, title, prompt, mistake, targets, new Dictionary<string, string[]> { [""] = targets }, false);

    private static Pose TwoSided(string id, string title, string prompt, string mistake, params string[] stems)
    {
        var targets = Both(stems);
        return new Pose(id, title, prompt, mistake, targets, new Dictionary<string, string[]>
        {
            ["L"] = targets.Where(t => t.EndsWith("Left")).ToArray(),
            ["R"] = targets.Where(t => t.EndsWith("Right")).ToArray(),
        }, false);
    }

    private static Pose SidewaysMove(string id, string title, string prompt, string mistake, string[] left, string[] right) =>
        new(id, title, prompt, mistake, left.Concat(right).ToArray(),
            new Dictionary<string, string[]> { ["L"] = left, ["R"] = right }, true);

    public static readonly Pose Relax = Whole("relax", "Relax",
        "Relax your face completely. Lips together, jaw loose.", "");

    public static readonly Pose RelaxEnd = Whole("relax_end", "Relax",
        "Relax. That was the last one.", "");

    public static readonly IReadOnlyList<Pose> All =
    [
        Relax,
        Whole("jaw_open", "Jaw open", "Open your jaw as wide as is comfortable.",
            "Keep it comfortable; no need to strain.", "JawOpen"),
        Whole("lips_pressed_open", "Lips pressed, jaw half open",
            "Open your jaw about halfway, and keep your lips stretched and pressed together.",
            "Open your jaw a bit more, and keep the lips sealed.", "MouthClosed"),
        TwoSided("funnel", "Funnel",
            "Push your lips forward and flare them out like a trumpet, open in the middle.",
            "Not a kiss: keep the middle open and the lips turned out.", "LipFunnelUpper", "LipFunnelLower"),
        TwoSided("pucker", "Pucker", "Pucker your lips, like a kiss.",
            "Push the lips forward; don't puff the cheeks.", "LipPuckerUpper", "LipPuckerLower"),
        TwoSided("lips_in", "Lips in", "Roll both lips into your mouth.",
            "Roll them in so the lips disappear.", "LipSuckUpper", "LipSuckLower"),
        TwoSided("cheek_puff", "Cheek puff", "Puff your cheeks full of air.",
            "Puff out; don't push the lips forward.", "CheekPuff"),
        TwoSided("cheek_suck", "Cheeks in", "Suck your cheeks in. Fish face.",
            "Pull the cheeks in between your teeth.", "CheekSuck"),
        TwoSided("smile", "Big smile", "Big smile. Show your teeth.",
            "A full smile, corners up.", "MouthCornerPull", "MouthCornerSlant", "CheekSquint"),
        TwoSided("upper_lip", "Upper lip up", "Raise your upper lip to show your top teeth.",
            "Lift the upper lip; don't wrinkle your nose.", "MouthUpperUp", "MouthUpperDeepen"),
        TwoSided("lower_lip", "Lower lip down", "Pull your lower lip down to show your bottom teeth.",
            "Pull the lower lip down; keep the jaw still.", "MouthLowerDown"),
        TwoSided("frown", "Frown", "Pull the corners of your mouth down. A sad frown.",
            "Corners down, lips closed.", "MouthFrown"),
        TwoSided("stretch", "Stretch",
            "Lips closed and flat, pull the corners of your mouth straight out toward your ears.",
            "Not a smile, and keep your teeth hidden.", "MouthStretch"),
        TwoSided("press", "Press", "Lips closed, press them firmly together, like holding back a laugh.",
            "Press them flat; don't roll them in.", "MouthPress", "MouthTightener"),
        Whole("chin_raise", "Pout", "Push your lower lip up and out. A pout.",
            "Push the lower lip up and out.", "MouthRaiserLower", "MouthRaiserUpper"),
        TwoSided("sneer", "Sneer", "Wrinkle your nose. A disgusted sneer.",
            "Wrinkle the nose hard.", "NoseSneer"),
        SidewaysMove("jaw_sideways", "Jaw left and right",
            "Move your jaw to one side and hold it, then to the other side.",
            "Move the whole jaw, not just the lips.", ["JawLeft"], ["JawRight"]),
        Whole("jaw_forward", "Jaw forward", "Lips relaxed, slide your lower jaw forward into an underbite.",
            "Move the jaw itself; don't pout.", "JawForward"),
        SidewaysMove("mouth_sideways", "Mouth left and right",
            "Lips together. Shift your whole mouth to one side and hold it, then to the other side.",
            "Keep the lips together and move the whole mouth.",
            ["MouthUpperLeft", "MouthLowerLeft"], ["MouthUpperRight", "MouthLowerRight"]),
        TwoSided("squint", "Squint", "Squint your eyes hard, but keep them open.",
            "Squint, but keep them open.", "EyeSquint"),
        TwoSided("eyes_wide", "Eyes wide", "Open your eyes as wide as you can.",
            "Open the eyes wide.", "EyeWide"),
        TwoSided("brows_up", "Eyebrows up", "Raise your eyebrows as high as you can.",
            "Raise both eyebrows high.", "BrowInnerUp", "BrowOuterUp"),
        TwoSided("brows_down", "Eyebrows down", "Pull your eyebrows down and together. Angry.",
            "Pull them down and together.", "BrowLowerer", "BrowPinch"),
        Whole("tongue_out", "Tongue out", "Stick your tongue out.", "Stick it out as far as is comfortable.",
            "TongueOut"),
        RelaxEnd,
    ];

    public static Pose Get(string id) => All.First(p => p.Id == id);

    /// <summary> The Quest Pro (Virtual Desktop) default run: every pose (DESIGN 8.1, nov-ash 2026-10-11). </summary>
    public static IReadOnlyList<Pose> DefaultRun(string? moduleName) => All;

    /// <summary> Sideways pairs that share one ceiling (poses.py SIDEWAYS). </summary>
    public static readonly (string A, string B)[] SidewaysPairs =
    [
        ("JawLeft", "JawRight"), ("MouthUpperLeft", "MouthUpperRight"), ("MouthLowerLeft", "MouthLowerRight"),
    ];
}
