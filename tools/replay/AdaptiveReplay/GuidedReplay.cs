using System.Text.Json;
using System.Text.Json.Nodes;
using VRCFaceTracking.Core.Adaptive.Calibration;
using VRCFaceTracking.Core.Adaptive.Guided;

/// <summary>
/// AdaptiveReplay guided MANIFEST.json DATA.bin OUT.json: drives a whole GuidedSession with recorded pose frames, as
/// if the person did each pose when asked (guided_replay.py). Each library pose plays its recorded window over and
/// over until the ring fills; when the session asks, the simulated person skips. Intro / "got it" play relax frames.
/// </summary>
internal static class GuidedReplay
{
    public static int Run(string[] args)
    {
        var manifest = JsonNode.Parse(File.ReadAllText(args[0]))!;
        var names = manifest["names"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        double[] Vec(JsonNode node) => node.AsArray().Select(v => v!.GetValue<double>()).ToArray();
        var k = names.Length;
        var frames = new Dictionary<string, List<float[]>>();
        using (var data = File.OpenRead(args[1]))
        using (var reader = new BinaryReader(data))
        {
            foreach (var (id, w) in manifest["windows"]!.AsObject())
            {
                data.Position = w!["offset"]!.GetValue<long>();
                var n = w["n"]!.GetValue<int>();
                var list = new List<float[]>();
                for (var f = 0; f < n; f++)
                {
                    var row = new float[k];
                    for (var i = 0; i < k; i++) row[i] = (float)reader.ReadDouble();
                    list.Add(row);
                }

                frames[id] = list;
            }
        }

        var priors = CalibrationPriors.Load(manifest["priors"]!.GetValue<string>()).Reorder(names);
        var ceilings = priors.C0.Select(c => Math.Min(Math.Max(c, 0.15), 1.0)).ToArray();
        var info = new GuidedInfo("Virtual Desktop", priors, ceilings, Vec(manifest["everyday"]!));
        var session = new GuidedSession(info, PoseLibrary.All, GuidedModuleData.ForModule("Virtual Desktop"),
            GuidedHistory.Empty);
        var spoken = new List<string>();
        GuidedResult? lastResult = null;
        session.Cue += c =>
        {
            if (c.Kind == GuidedCueKind.Speak) spoken.Add(c.Text!);
        };
        session.Start();

        var t = 0.0;
        var cursor = 0;
        string? playing = null;
        var redo = new Queue<string>(manifest["redo"]?.AsArray().Select(r => r!.GetValue<string>()) ?? []);
        var summaries = new JsonArray();
        for (var step = 0; step < 400_000; step++)
        {
            var snap = session.Snapshot;
            if (snap.Phase == GuidedPhase.Summary)
            {
                if (summaries.Count == 0 || !ReferenceEquals(snap.Result, lastResult))
                {
                    lastResult = snap.Result;
                    summaries.Add(new JsonArray(snap.Result!.Summary
                        .Select(s => (JsonNode)$"{s.Title}: {s.Change} ({s.Status}, {s.Color})").ToArray()));
                }

                if (redo.Count == 0)
                {
                    break;
                }

                session.Redo(redo.Dequeue()); // "Redo (optional)" on the summary
                continue;
            }

            if (snap.Phase == GuidedPhase.Ask)
            {
                session.Skip();
                continue;
            }

            var want = snap.Phase is GuidedPhase.Hold or GuidedPhase.Relax && snap.Pose != null ? snap.Pose.Id : "relax";
            if (!frames.ContainsKey(want)) want = "relax";
            if (want != playing)
            {
                playing = want;
                cursor = 0;
            }

            var src = frames[playing];
            t += 0.01;
            session.Feed(t, src[cursor % src.Count]);
            cursor++;
        }

        var result = session.Snapshot.Result;
        if (result == null)
        {
            Console.Error.WriteLine("the session did not finish");
            return 1;
        }

        var outJson = new JsonObject
        {
            ["seconds"] = t,
            ["poses"] = new JsonArray(result.Poses.Select(p => (JsonNode)new JsonObject
            {
                ["id"] = p.Id, ["done"] = p.Done, ["skipped"] = p.Skipped, ["filled_s"] = p.FilledS, ["weak"] = p.Weak,
                ["asks"] = new JsonArray(p.Asks.Select(a => (JsonNode)a).ToArray()),
            }).ToArray()),
            ["shapes"] = new JsonObject(result.Shapes.Select(s => KeyValuePair.Create(s.Name, (JsonNode?)new JsonObject
            {
                ["c"] = s.Ceiling, ["basis"] = s.Basis,
            }))),
            ["coupling"] = result.Coupling,
            ["priors_json"] = result.NewPriors.ToJson(),
            ["summary"] = new JsonArray(result.Summary.Select(s => (JsonNode)$"{s.Title}: {s.Change} ({s.Status}, {s.Color})").ToArray()),
            ["summaries"] = summaries,
            ["spoken"] = new JsonArray(spoken.Select(s => (JsonNode)s).ToArray()),
        };
        File.WriteAllText(args[2], outJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
