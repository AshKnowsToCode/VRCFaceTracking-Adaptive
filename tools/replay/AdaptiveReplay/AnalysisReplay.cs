using System.Text.Json.Nodes;
using VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// For golden_analysis.py:
///   AdaptiveReplay measure MANIFEST.json DATA.bin OUT.json   - PoseMeasure on each pose window's frames
///   AdaptiveReplay analyze IN.json OUT.json                  - PoseAnalysis on measured runs
/// </summary>
internal static class AnalysisReplay
{
    public static int Measure(string[] args)
    {
        var manifest = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var names = manifest["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        var dt = manifest["dt"]!.GetValue<double>();
        var k = names.Length;
        using var data = File.OpenRead(args[1]);
        using var reader = new BinaryReader(data);
        var sessions = new JsonArray();
        foreach (var s in manifest["sessions"]!.AsArray())
        {
            var held = new JsonObject();
            var rest = new List<double[]>();
            JsonNode? ratio = null;
            foreach (var p in s!["poses"]!.AsArray())
            {
                var id = p!["id"]!.GetValue<string>();
                var relax = p["relax"]!.GetValue<bool>();
                var m = new PoseMeasure(id, names, relax);
                data.Position = p["offset"]!.GetValue<long>();
                var n = p["n"]!.GetValue<int>();
                var x = new double[k];
                for (var f = 0; f < n; f++)
                {
                    for (var i = 0; i < k; i++) x[i] = reader.ReadDouble();
                    m.Update(dt, x);
                }

                if (m.Held() is not { } h)
                {
                    continue; // poses.py skips poses with under 0.5 s of frames
                }

                held[id] = new JsonArray(h.Select(v => (JsonNode)v).ToArray());
                if (relax) rest.AddRange(m.RestFrames);
                if (m.Ratio() is { } r) ratio = new JsonArray(r.Ratio, r.Jaw);
            }

            var p95 = Enumerable.Range(0, k).Select(i => Stats.Percentile(rest.Select(r => r[i]).ToList(), 95)).ToArray();
            sessions.Add(new JsonObject
            {
                ["held"] = held,
                ["rest_p95"] = new JsonArray(p95.Select(v => (JsonNode)v).ToArray()),
                ["ratio"] = ratio,
            });
        }

        File.WriteAllText(args[2], new JsonObject { ["sessions"] = sessions }.ToJsonString());
        return 0;
    }

    public static int Analyze(string[] args)
    {
        var input = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var names = input["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        double[] Vec(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();
        var poses = input["poses"]!.AsArray()
            .Select(p => (p!["id"]!.GetValue<string>(), p["targets"]!.AsArray().Select(t => t!.GetValue<string>()).ToArray()))
            .ToList();
        var runs = input["runs"]!.AsArray().Select(r => new RunMeasurement
        {
            Held = r!["held"]!.AsObject().ToDictionary(kv => kv.Key, kv => Vec(kv.Value!)),
            RestP95 = Vec(r["rest_p95"]!),
            LipsRatio = r["ratio"] is JsonArray a ? (a[0]!.GetValue<double>(), a[1]!.GetValue<double>()) : null,
        }).ToList();

        var (shapes, coupling) = PoseAnalysis.Analyze(names, poses, runs, Vec(input["now"]!), Vec(input["everyday"]!));
        var outShapes = new JsonObject();
        foreach (var s in shapes)
        {
            outShapes[s.Name] = new JsonObject { ["c"] = s.Ceiling, ["basis"] = s.Basis };
        }

        File.WriteAllText(args[1], new JsonObject { ["shapes"] = outShapes, ["coupling"] = coupling }.ToJsonString());
        return 0;
    }
}
