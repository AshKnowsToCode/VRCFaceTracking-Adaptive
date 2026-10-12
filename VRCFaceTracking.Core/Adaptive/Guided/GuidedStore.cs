using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using VRCFaceTracking.Core.Adaptive.Calibration;
using VRCFaceTracking.Core.Adaptive.Recording;

namespace VRCFaceTracking.Core.Adaptive.Guided;

/// <summary> A saved guided run, as the history list shows it. </summary>
public sealed record GuidedReportInfo(string Path, DateTime Started, string Module, bool Kept, int Measured, int Skipped,
    int Asked);

/// <summary>
/// Guided calibration reports: every finished run is saved (kept or discarded) under
/// <c>%APPDATA%\VRCFaceTracking-Adaptive\adaptive-calibration\guided\</c>, numbers only, kept indefinitely (DESIGN 8.1
/// "report history"). Kept runs feed later runs: their measurements are combined with the next one and their best
/// holds become the ring's "your best in this pose".
/// </summary>
public static class GuidedStore
{
    public const int CombineRuns = 2; // earlier kept runs combined with a new one

    public static string Folder => Path.Combine(AdaptiveCalibration.StateDirectory, "guided");

    public static string SaveReport(GuidedResult r, bool kept)
    {
        Directory.CreateDirectory(Folder);
        var held = new JsonObject();
        foreach (var (pose, values) in r.Run.Held)
        {
            held[pose] = Arr(values);
        }

        var index = r.Names.Select((s, i) => (s, i)).ToDictionary(p => p.s, p => p.i);
        var report = new JsonObject
        {
            ["schema"] = 1,
            ["started"] = r.Started.ToString("O"),
            ["saved"] = DateTime.Now.ToString("O"),
            ["module"] = r.ModuleKey,
            ["kept"] = kept,
            ["runs_combined"] = r.RunsCombined,
            ["expressions"] = new JsonArray(r.Names.Select(n => (JsonNode)n).ToArray()),
            ["poses"] = new JsonArray(r.Poses.Select(p => (JsonNode)new JsonObject
            {
                ["id"] = p.Id,
                ["done"] = p.Done,
                ["skipped"] = p.Skipped,
                ["asks"] = new JsonArray(p.Asks.Select(a => (JsonNode)a).ToArray()),
                ["filled_s"] = p.FilledS,
                ["weak"] = p.Weak,
                ["recorded"] = new JsonObject(p.Recorded.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
            }).ToArray()),
            ["measurement"] = new JsonObject
            {
                ["held"] = held,
                ["rest_p95"] = Arr(r.Run.RestP95),
                ["lips_ratio"] = r.Run.LipsRatio is { } lr ? new JsonArray(lr.Ratio, lr.Jaw) : null,
            },
            ["shapes"] = new JsonArray(r.Shapes.Select(s => (JsonNode)new JsonObject
            {
                ["name"] = s.Name,
                ["before"] = r.Before[index[s.Name]],
                ["after"] = s.Ceiling,
                ["basis"] = s.Basis,
                ["measured_runs"] = s.MeasuredRuns,
                ["spread"] = s.Spread,
            }).ToArray()),
            ["coupling"] = r.Coupling,
            ["summary"] = new JsonArray(r.Summary.Select(s => (JsonNode)new JsonObject
            {
                ["pose"] = s.PoseId,
                ["title"] = s.Title,
                ["change"] = s.Change,
                ["status"] = s.Status,
                ["color"] = s.Color.ToString(),
            }).ToArray()),
        };

        var path = Path.Combine(Folder, $"run-{r.Started:yyyyMMdd-HHmmss}.json");
        File.WriteAllText(path, report.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), new UTF8Encoding(false));
        SessionRecorder.LogEvent("guided", (kept ? "keep " : "discard ") + Path.GetFileName(path));
        return path;
    }

    /// <summary> Earlier kept runs for this module, newest first: measurements to combine and best holds. </summary>
    public static GuidedHistory LoadHistory(string moduleKey, IReadOnlyList<string> names)
    {
        var history = new GuidedHistory();
        foreach (var info in ListReports().Where(r => r.Kept && r.Module == moduleKey))
        {
            try
            {
                var d = JsonNode.Parse(File.ReadAllText(info.Path))!;
                var expr = d["expressions"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
                if (!expr.SequenceEqual(names))
                {
                    continue; // another expression set (older app version)
                }

                if (history.Runs.Count < CombineRuns)
                {
                    var m = d["measurement"]!;
                    history.Runs.Add(new RunMeasurement
                    {
                        Held = m["held"]!.AsObject().ToDictionary(kv => kv.Key, kv => Vec(kv.Value!)),
                        RestP95 = Vec(m["rest_p95"]!),
                        LipsRatio = m["lips_ratio"] is JsonArray a ? (a[0]!.GetValue<double>(), a[1]!.GetValue<double>()) : null,
                    });
                }

                foreach (var p in d["poses"]!.AsArray())
                {
                    var best = history.Best.TryGetValue(p!["id"]!.GetValue<string>(), out var b)
                        ? b
                        : history.Best[p["id"]!.GetValue<string>()] = new Dictionary<string, double>();
                    foreach (var (shape, v) in p["recorded"]!.AsObject())
                    {
                        var value = v!.GetValue<double>();
                        if (!best.TryGetValue(shape, out var old) || value > old)
                        {
                            best[shape] = value;
                        }
                    }
                }
            }
            catch (Exception)
            {
                // an unreadable report is skipped, never fatal
            }
        }

        return history;
    }

    public static List<GuidedReportInfo> ListReports()
    {
        var list = new List<GuidedReportInfo>();
        if (!Directory.Exists(Folder))
        {
            return list;
        }

        foreach (var path in Directory.GetFiles(Folder, "run-*.json"))
        {
            try
            {
                var d = JsonNode.Parse(File.ReadAllText(path))!;
                var poses = d["poses"]!.AsArray();
                list.Add(new GuidedReportInfo(path, DateTime.Parse(d["started"]!.GetValue<string>()),
                    d["module"]!.GetValue<string>(), d["kept"]!.GetValue<bool>(),
                    poses.Count(p => p!["done"]!.GetValue<bool>() && !p["skipped"]!.GetValue<bool>()),
                    poses.Count(p => p!["skipped"]!.GetValue<bool>()),
                    poses.Sum(p => p!["asks"]!.AsArray().Count)));
            }
            catch (Exception)
            {
                // skip unreadable files
            }
        }

        return list.OrderByDescending(r => r.Started).ToList();
    }

    private static JsonArray Arr(IEnumerable<double> values) => new(values.Select(v => (JsonNode)v).ToArray());

    private static double[] Vec(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();
}
