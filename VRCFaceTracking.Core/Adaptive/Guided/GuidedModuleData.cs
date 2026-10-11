using System.Reflection;
using Newtonsoft.Json.Linq;

namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// Per-module pose data for the guided calibration (built by tools/replay/guided_module_data.py from recorded pose
/// sessions): per pose the expressions that move with it (never "wrong expression") and a typical "best in this pose"
/// per asked-for expression, the ring's reference on a person's first run. Unknown module: empty (no companions,
/// references fall back to the expression's ceiling).
/// </summary>
public sealed class GuidedModuleData
{
    private const string ResourcePrefix = "VRCFaceTracking.Core.Adaptive.Guided.Data.";

    private static readonly Dictionary<string, string> ModuleFiles = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Virtual Desktop"] = "guided-virtual-desktop.json",
    };

    public static readonly GuidedModuleData Empty = new();

    public Dictionary<string, HashSet<string>> Companions { get; } = new();

    public Dictionary<string, Dictionary<string, double>> Typical { get; } = new();

    public static GuidedModuleData ForModule(string? moduleName)
    {
        if (moduleName == null || !ModuleFiles.TryGetValue(moduleName, out var file))
        {
            return Empty;
        }

        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourcePrefix + file);
        if (stream == null)
        {
            return Empty;
        }

        using var reader = new StreamReader(stream);
        var d = JObject.Parse(reader.ReadToEnd());
        var data = new GuidedModuleData();
        foreach (var (pose, v) in (JObject)d["poses"]!)
        {
            data.Companions[pose] = v!["companions"]!.Values<string>().Select(s => s!).ToHashSet();
            data.Typical[pose] = ((JObject)v["typical"]!).Properties().ToDictionary(p => p.Name, p => p.Value.Value<double>());
        }

        return data;
    }
}
