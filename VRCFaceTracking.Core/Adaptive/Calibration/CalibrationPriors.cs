using System.Reflection;
using Newtonsoft.Json.Linq;

namespace VRCFaceTracking.Core.Adaptive.Calibration;

/// <summary>
/// Starting values for one tracking module: per expression the ceiling (raw value that maps to 100%), rest level and
/// noise, plus coupled shapes. Same JSON as tools/replay/priors/*.json (built by priors.py / poses.py). An unknown
/// module gets <see cref="Identity"/>, which behaves like calibration off plus a small deadzone.
/// </summary>
public sealed class CalibrationPriors
{
    public required string Name { get; init; }
    public required string[] Expressions { get; init; }
    /// <summary> Ceiling prior. </summary>
    public required double[] C0 { get; init; }
    /// <summary> Rest prior. </summary>
    public required double[] B0 { get; init; }
    /// <summary> Noise prior. </summary>
    public required double[] Sigma0 { get; init; }
    /// <summary> Child -> (parent, ratio): the child takes its parent's scale times the ratio (DESIGN 6.10). </summary>
    public required IReadOnlyDictionary<string, (string Parent, double Ratio)> Coupled { get; init; }

    /// <summary>
    /// Optional per-expression upper limit for the learned ceiling, above c0 x 1.6 (0 = none). A guided calibration
    /// that makes an expression stronger stores the ceiling in effect before it, so normal use can calm it all the way
    /// back if everyday expressions go past the new value (adaptive_v0.Priors.c_max).
    /// </summary>
    public double[]? CMax { get; init; }

    private const string ResourcePrefix = "VRCFaceTracking.Core.Adaptive.Calibration.Priors.";

    /// <summary> Embedded priors by tracking module name (<c>ModuleMetadata.Name</c>). </summary>
    private static readonly Dictionary<string, string> ModulePriors = new(StringComparer.OrdinalIgnoreCase)
    {
        // Quest Pro via Virtual Desktop. Measured on one person (nov-ash, two pose sessions 2026-10-10); a start
        // point until volunteer data gives a per-module average (DESIGN 8.4).
        ["Virtual Desktop"] = "virtual-desktop.json",
    };

    public static CalibrationPriors Parse(string json, string fallbackName)
    {
        var d = JObject.Parse(json);
        var coupled = new Dictionary<string, (string, double)>();
        if (d["coupled"] is JObject c)
        {
            foreach (var (child, v) in c)
            {
                if (v?["parent"]?.Value<string>() is { } parent && v["ratio"] is { } ratio)
                {
                    coupled[child] = (parent, ratio.Value<double>());
                }
            }
        }

        return new CalibrationPriors
        {
            Name = d["meta"]?["name"]?.Value<string>() ?? fallbackName,
            Expressions = d["expressions"]!.Values<string>().Select(s => s!).ToArray(),
            C0 = d["c0"]!.Values<double>().ToArray(),
            B0 = d["b0"]!.Values<double>().ToArray(),
            Sigma0 = d["sigma0"]!.Values<double>().ToArray(),
            Coupled = coupled,
            CMax = d["c_max"] is JArray cm ? cm.Values<double>().ToArray() : null,
        };
    }

    public static CalibrationPriors Load(string path) => Parse(File.ReadAllText(path), Path.GetFileNameWithoutExtension(path));

    /// <summary> Where they came from, written into the JSON's meta (e.g. which guided runs). </summary>
    public string? Source { get; init; }

    /// <summary> Same JSON layout as tools/replay/priors/*.json. </summary>
    public string ToJson()
    {
        var coupled = new JObject();
        foreach (var (child, (parent, ratio)) in Coupled)
        {
            coupled[child] = new JObject { ["parent"] = parent, ["ratio"] = Math.Round(ratio, 4) };
        }

        return new JObject
        {
            ["schema"] = 1,
            ["meta"] = new JObject { ["name"] = Name, ["source"] = Source ?? "" },
            ["expressions"] = new JArray(Expressions),
            ["c0"] = new JArray(C0.Select(v => Math.Round(v, 6))),
            ["b0"] = new JArray(B0.Select(v => Math.Round(v, 6))),
            ["sigma0"] = new JArray(Sigma0.Select(v => Math.Round(v, 6))),
            ["coupled"] = coupled,
            ["c_max"] = CMax != null ? new JArray(CMax.Select(v => Math.Round(v, 6))) : null,
        }.ToString(Newtonsoft.Json.Formatting.Indented);
    }

    public static CalibrationPriors Identity(IReadOnlyList<string> names) => new()
    {
        Name = "identity",
        Expressions = names.ToArray(),
        C0 = Enumerable.Repeat(1.0, names.Count).ToArray(),
        B0 = new double[names.Count],
        Sigma0 = Enumerable.Repeat(0.005, names.Count).ToArray(),
        Coupled = new Dictionary<string, (string, double)> { ["MouthClosed"] = ("JawOpen", 1.0) },
    };

    /// <summary> The embedded priors for a module, or null when there are none for it. </summary>
    public static CalibrationPriors? ForModule(string moduleName)
    {
        if (!ModulePriors.TryGetValue(moduleName, out var file))
        {
            return null;
        }

        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourcePrefix + file);
        if (stream == null)
        {
            return null;
        }

        using var reader = new StreamReader(stream);
        return Parse(reader.ReadToEnd(), Path.GetFileNameWithoutExtension(file));
    }

    /// <summary> The same priors in another expression order; names it lacks get identity values. </summary>
    public CalibrationPriors Reorder(IReadOnlyList<string> names)
    {
        var index = new Dictionary<string, int>();
        for (var i = 0; i < Expressions.Length; i++)
        {
            index[Expressions[i]] = i;
        }

        double[] Pick(double[] a, double fallback) =>
            names.Select(n => index.TryGetValue(n, out var i) ? a[i] : fallback).ToArray();

        return new CalibrationPriors
        {
            Name = Name,
            Expressions = names.ToArray(),
            C0 = Pick(C0, 1.0),
            B0 = Pick(B0, 0.0),
            Sigma0 = Pick(Sigma0, 0.005),
            Coupled = Coupled,
            CMax = CMax != null ? Pick(CMax, 0.0) : null,
            Source = Source,
        };
    }
}
