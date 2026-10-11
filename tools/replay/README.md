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

Calibration prototype (v0, DESIGN.md section 10):

```
python priors.py --known-good LocalSettings.json --out priors/NAME.json REC.vftrec ...   # build v0 priors
python evaluate.py --priors priors/NAME.json --known-good LocalSettings.json [--loops N] [--spikes] REC.vftrec ...
```

- `stock.py` - exact emulators of the stock Calibration and Unified Correctors (vectorised). Replaying
  `20261008-214509.vftrec` through them reproduces its out block for all 88 shapes to 1e-7 (resets at 40.7, 44.9,
  84.8 s, `warm=True`).
- `adaptive_v0.py` - the v0 calibrator: `AdaptiveCalibrator(Priors).step(t, x) -> y`. Golden reference for the C#
  port (`VRCFaceTracking.Core/Adaptive/Calibration/AdaptiveCalibrator.cs`); change both together.
- `golden_v0.py` - golden test of the C# port: `python golden_v0.py REC.vftrec ... [--priors P] [--module NAME]
  [--split 0.5]` builds `AdaptiveReplay/` (a console that replays a .vftrec through the C# calibrator), runs both and
  requires every output, the per-frame gate and the final ceilings to match (tolerance 1e-4; bit-exact so far).
  `--module "Virtual Desktop"` uses the priors embedded in the app; `--split` saves the state as JSON mid-way and
  continues on a fresh calibrator in both languages.
- `priors.py` - ceiling priors from a trusted stock calibration (floored at 0.15), rest/noise priors from
  recordings, the MouthClosed -> JawOpen coupling ratio.
- `evaluate.py` - candidates raw / stock_known_good / stock_reset / v0 / v0_identity / v0_noguard. Metrics: lips
  sealed, pinned during speech (mic level > -60 dBFS), max gain, rest jitter, p95 / p99.5 per group vs now;
  `--loops` = hours-of-use drift, `--spikes` = robustness to one-frame spikes.
- `poses.py` - pose session (manual stand-in for the guided boost, DESIGN 8.1): `guide` builds a spoken audio
  guide with the Windows SAPI voice (`tts_sapi.vbs`) plus its timeline `pose_guide_vN.json`; `analyze` finds the
  guide in a recording by matching the whole timeline (or uses one F8 marker per pose), measures each shape's
  highest level held >= 0.5 s, the MouthClosed / JawOpen ratio, cross-talk and L/R differences, compares against the
  old values with the avatar settings each shape feeds, and writes measured priors; `selftest` checks it on
  synthetic sessions. Person-facing checklist: `docs/adaptive-calibration/POSE_SESSION.md`.
- `ring_sim.py` - the guided boost's fill ring (DESIGN 8.1): when a pose counts, how fast the ring fills, what it
  records, and when it stops to ask (retry or skip). Reference for the C# page. `stress REC.vftrec --report OUT.md`
  replays recorded pose sessions through the first-sketch and revised rules, as recorded and with simulated
  problems (headset reading 50% / 30%, tremor, one side not moving).

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
