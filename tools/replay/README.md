# Replay tools (adaptive calibration)

Python tools for the session recordings made by the fork's **Session Recorder** mutation
(`VRCFaceTracking.Core/Adaptive/Recording`). Recordings are written to
`<recordings folder>\YYYYMMDD-HHMMSS.vftrec` and never leave the PC. The folder is a
recorder setting; blank means `Documents\VRCFaceTracking-Adaptive\recordings`.

Needs Python 3.10+ with `pip install -r requirements.txt` (numpy, matplotlib).

```
python vftrec.py info FILE [--session] [--shapes]      # summary: rate, markers, events, mic level, per-shape stats
python plot_vftrec.py FILE [--session] [--shapes JawOpen LipPucker] [--start S --end S] [--out plot.png]
```

`--session` joins every file of the same app run (hourly rotation, idle gaps). A shape name that is not an
expression averages the expressions starting with it (`LipPucker` = the four pucker shapes). Files that are still
being written, or were cut off by a crash, read fine up to the last flush (about 2 s); `info` marks them truncated.

In code:

```python
from vftrec import read, read_session
rec = read("20261007-201500.vftrec")
rec.t                      # seconds since session start, per frame
rec.raw_col("JawOpen")     # module output, before any mutation
rec.out_col("JawOpen")     # after all mutations (what VRChat gets, before OSC)
rec.audio_t, rec.audio_db  # mic RMS dBFS per 10 ms
rec.markers                # [(tick, label)]
rec.events_of("tracking")  # [(seconds, "zeros" | "frozen" | "ok")]
```

## Format (schema 1)

Defined in `RecordingFormat.cs`; bump the schema there and in `vftrec.py` together. One UTF-8 JSON header line,
then a raw deflate stream of little-endian records `type:u8, tick:i64 (QPC), payload`:

| Type | Payload |
|------|---------|
| `F` frame | `float32[2 * block_length]`: raw block then out block. Block = 14 eye/head channels + 88 expressions in enum order |
| `A` audio | `float32` RMS dBFS of one 10 ms mic block, tick = end of block (WASAPI capture time). Silence = -120 |
| `M` marker | `uint16 n, utf8[n]` label (UI button or F8 in the VRCFT window) |
| `G` event | `uint16 n, utf8[n]` `kind:detail` |

Event kinds: `tracking:zeros|frozen|ok` (all-zero frames, or no shape moving > 1e-4 for 0.5 s), `focus:vrchat|vrcft|other|none`
(foreground app, category only), `modules:[...]`, `audio:device ...|on ...|off|no_device|device_changed|error ...`,
`recorder:dropped N|stop <exit|rotate|idle|disabled|error>`.

The header holds the QPC frequency, session id and file index, a QPC/UTC clock pair, module names, expression and
channel names, and a settings snapshot of every mutation in run order (UI fields including statics, plus the saved
state, e.g. the stock Calibration's learned maxima).
