using System.Text.Json;
using System.Text.Json.Nodes;
using VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// AdaptiveReplay ring MANIFEST.json DATA.bin OUT.json: runs every pose case written by golden_ring.py through the
/// C# <see cref="PoseRing"/> and writes each half's result. DATA.bin holds each case's frames as float64 rows.
/// </summary>
internal static class RingReplay
{
    public static int Run(string[] args)
    {
        if (args.Length != 3)
        {
            Console.Error.WriteLine("usage: AdaptiveReplay ring MANIFEST.json DATA.bin OUT.json");
            return 2;
        }

        var manifest = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var names = manifest["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        var dt = manifest["dt"]!.GetValue<double>();
        var k = names.Length;
        using var data = File.OpenRead(args[1]);
        using var reader = new BinaryReader(data);
        var results = new JsonArray();
        foreach (var c in manifest["cases"]!.AsArray())
        {
            var halves = c!["halves"]!.AsObject().ToDictionary(
                kv => kv.Key, kv => kv.Value!.AsArray().Select(s => s!.GetValue<string>()).ToArray());
            var companions = c["companions"]!.AsArray().Select(s => s!.GetValue<string>()).ToHashSet();
            double[] Vec(string key) => c[key]!.AsArray().Select(v => v!.GetValue<double>()).ToArray();
            var jaw = c["jaw_raw"] is { } j ? j.GetValue<double>() : (double?)null;
            var ring = new PoseRing(c["pose"]!.GetValue<string>(), names, halves, companions, Vec("ref"), Vec("tref"),
                Vec("floor"), jaw);

            data.Position = c["offset"]!.GetValue<long>();
            var n = c["n"]!.GetValue<int>();
            var x = new double[k];
            for (var f = 0; f < n; f++)
            {
                for (var i = 0; i < k; i++)
                {
                    x[i] = reader.ReadDouble();
                }

                ring.Update(dt, x);
            }

            ring.Finish();
            var outHalves = new JsonArray();
            foreach (var h in ring.Halves)
            {
                var (need, reason) = ring.Summarize(h, dt);
                outHalves.Add(new JsonObject
                {
                    ["name"] = h.Name,
                    ["filled_s"] = h.FilledS,
                    ["fill"] = h.Fill,
                    ["need_s"] = need,
                    ["reason"] = reason,
                    ["weak"] = h.Weak,
                    ["level"] = h.Level,
                    ["rec_ratio"] = h.RecordedRatio,
                });
            }

            results.Add(new JsonObject { ["id"] = c["id"]!.GetValue<int>(), ["halves"] = outHalves });
        }

        File.WriteAllText(args[2], new JsonObject { ["cases"] = results }.ToJsonString());
        Console.WriteLine($"{results.Count} ring cases");
        return 0;
    }
}
