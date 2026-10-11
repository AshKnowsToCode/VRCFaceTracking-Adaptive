// Replays one .vftrec (schema 1) through the C# AdaptiveCalibrator and writes its outputs for golden_v0.py:
//   AdaptiveReplay --priors PRIORS.json --out OUT.bin [--split FRAME] REC.vftrec
// --split saves the learned state as JSON before FRAME and continues on a fresh calibrator loaded from it.
// OUT.bin (little-endian): int32 n, int32 k, float64 y[n*k], uint8 gated[n], float64 c[k], float64 g[k],
// float64 active_s.
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using VRCFaceTracking.Core.Adaptive.Calibration;

switch (args.Length > 0 ? args[0] : "")
{
    case "ring": return RingReplay.Run(args[1..]); // golden_ring.py
    case "measure": return AnalysisReplay.Measure(args[1..]); // golden_analysis.py
    case "analyze": return AnalysisReplay.Analyze(args[1..]); // golden_analysis.py
    case "guided": return GuidedReplay.Run(args[1..]); // guided_replay.py
    case "pictures": return PictureSheet.Run(args[1..]); // pose pictures for review
}

string? priorsPath = null, outPath = null, recPath = null, module = null;
var split = -1;
for (var a = 0; a < args.Length; a++)
{
    switch (args[a])
    {
        case "--priors": priorsPath = args[++a]; break;
        case "--module": module = args[++a]; break; // the priors embedded in the app for this module name
        case "--out": outPath = args[++a]; break;
        case "--split": split = int.Parse(args[++a]); break;
        default: recPath = args[a]; break;
    }
}

if ((priorsPath == null && module == null) || outPath == null || recPath == null)
{
    Console.Error.WriteLine("usage: AdaptiveReplay (--priors PRIORS.json | --module NAME) --out OUT.bin [--split FRAME] REC.vftrec");
    return 2;
}

var rec = Vftrec.Read(recPath);
var k = rec.Expressions.Length;
var loaded = module != null ? CalibrationPriors.ForModule(module) : CalibrationPriors.Load(priorsPath!);
if (loaded == null)
{
    Console.Error.WriteLine($"no embedded priors for module '{module}'");
    return 1;
}

var priors = loaded.Reorder(rec.Expressions);
var cal = new AdaptiveCalibrator(priors);
var n = rec.T.Length;
var y = new double[n * k];
var gated = new byte[n];
for (var f = 0; f < n; f++)
{
    if (f == split)
    {
        var json = JsonSerializer.Serialize(cal.SaveState());
        cal = new AdaptiveCalibrator(priors);
        if (!cal.LoadState(JsonSerializer.Deserialize<CalibrationState>(json)!))
        {
            Console.Error.WriteLine("state did not load");
            return 1;
        }
    }

    cal.Step(rec.T[f], rec.Raw.AsSpan(f * k, k), y.AsSpan(f * k, k));
    gated[f] = cal.Gated ? (byte)1 : (byte)0;
}

using (var w = new BinaryWriter(File.Create(outPath)))
{
    w.Write(n);
    w.Write(k);
    foreach (var v in y) w.Write(v);
    w.Write(gated);
    for (var i = 0; i < k; i++) w.Write(cal.Ceiling(i));
    for (var i = 0; i < k; i++) w.Write(cal.Guard(i));
    w.Write(cal.ActiveSeconds);
}

Console.WriteLine($"{Path.GetFileName(recPath)}: {n} frames, {cal.ActiveSeconds:0.0} active s");
return 0;

/// <summary> Minimal schema-1 reader: frame times and the raw expression block (see RecordingFormat, vftrec.py). </summary>
internal sealed class Vftrec
{
    public required double[] T;
    public required float[] Raw; // [n * expressions]
    public required string[] Expressions;

    public static Vftrec Read(string path)
    {
        var data = File.ReadAllBytes(path);
        var nl = Array.IndexOf(data, (byte)'\n');
        using var header = JsonDocument.Parse(Encoding.UTF8.GetString(data, 0, nl));
        var h = header.RootElement;
        if (h.GetProperty("schema").GetInt32() != 1)
        {
            throw new InvalidDataException("schema not supported");
        }

        var freq = h.GetProperty("qpc_frequency").GetInt64();
        var origin = h.GetProperty("session").GetProperty("qpc_start").GetInt64();
        var frame = h.GetProperty("frame");
        var blockLength = frame.GetProperty("block_length").GetInt32();
        var channels = frame.GetProperty("channels").GetArrayLength();
        var expressions = h.GetProperty("expressions").EnumerateArray().Select(e => e.GetString()!).ToArray();

        var times = new List<double>();
        var raw = new List<float>();
        using var body = new DeflateStream(new MemoryStream(data, nl + 1, data.Length - nl - 1), CompressionMode.Decompress);
        var buf = new byte[8 + 8 * blockLength];
        try
        {
            while (true)
            {
                var kind = body.ReadByte();
                if (kind < 0)
                {
                    break;
                }

                switch ((char)kind)
                {
                    case 'F':
                        if (!Fill(body, buf, 8 + 8 * blockLength)) return Done();
                        var tick = BitConverter.ToInt64(buf, 0);
                        times.Add((tick - origin) / (double)freq);
                        for (var i = 0; i < expressions.Length; i++)
                        {
                            raw.Add(BitConverter.ToSingle(buf, 8 + 4 * (channels + i)));
                        }

                        break;
                    case 'A':
                        if (!Fill(body, buf, 12)) return Done();
                        break;
                    case 'M':
                    case 'G':
                        if (!Fill(body, buf, 10)) return Done();
                        var len = BitConverter.ToUInt16(buf, 8);
                        var text = new byte[len];
                        if (!Fill(body, text, len)) return Done();
                        break;
                    default:
                        throw new InvalidDataException($"unknown record type {kind:x2}");
                }
            }
        }
        catch (InvalidDataException) when (times.Count > 0)
        {
            // cut off mid-stream (app killed or still recording): keep every complete frame
        }

        return Done();

        Vftrec Done() => new() { T = times.ToArray(), Raw = raw.ToArray(), Expressions = expressions };
    }

    private static bool Fill(Stream s, byte[] buf, int count)
    {
        var got = 0;
        while (got < count)
        {
            var r = s.Read(buf, got, count - got);
            if (r <= 0)
            {
                return false;
            }

            got += r;
        }

        return true;
    }
}
