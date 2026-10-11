"""Replay recordings through calibration candidates and score them (DESIGN.md section 5 metrics, v0 subset).

Candidates (each followed by the stock Unified Correctors, as in the live pipeline):
  raw                calibration off
  stock_known_good   stock Calibration with a trusted LocalSettings.json (what nov-ash runs now)
  stock_reset        stock Calibration right after "Reset Calibration" (learns from the recording itself)
  v0                 adaptive_v0 with the given priors
  v0_identity        adaptive_v0 with identity priors (an unknown module)
  v0_noguard         adaptive_v0 with the saturation guardrail off

Stress tests: --loops N replays the recordings N times with persistent state (hours of use); --spikes injects
one-frame 1.0 spikes into random mouth shapes every 5 s.

    python evaluate.py --priors priors/vd_questpro_novash_v0.json --known-good LocalSettings.json REC.vftrec ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adaptive_v0 import AdaptiveCalibrator, Params, Priors  # noqa: E402
from priors import worn_mask  # noqa: E402
from stock import StockCalibration, correctors, load_calibration  # noqa: E402
from vftrec import read  # noqa: E402

MOUTH = ("Jaw", "Mouth", "Lip", "Cheek", "Nose")
GROUPS = {"jaw open": ["JawOpen"], "lips pressed": ["MouthClosed"], "smile": ["MouthCornerPull"],
          "upper lip up": ["MouthUpperUp"], "lower lip down": ["MouthLowerDown"], "funnel": ["LipFunnel"],
          "pucker": ["LipPucker"], "lip suck": ["LipSuck"], "frown": ["MouthFrown"], "cheek puff": ["CheekPuff"],
          "mouth sideways": ["MouthUpperLeft", "MouthUpperRight"], "stretch": ["MouthStretch"]}
SPEECH_DB = -60.0  # nov-ash's ModMic: background -75..-90 dBFS, voice -60..-35


def speech_mask(rec, t, window=0.15):
    at, db = rec.audio_t, rec.audio_db
    loud = np.r_[0, np.cumsum(db > SPEECH_DB)]
    lo, hi = np.searchsorted(at, t - window), np.searchsorted(at, t + window)
    return (loud[hi] - loud[lo]) > 0


class Candidate:
    def __init__(self, name, make):
        self.name, self.make = name, make
        self.cal = None

    def run(self, t, X, names, persistent=False, t_offset=0.0):
        if self.cal is None or not persistent:
            self.cal = self.make()
        t = t + t_offset  # persistent runs need a clock that keeps going forward
        if isinstance(self.cal, AdaptiveCalibrator):
            f = lambda i: self.cal.step(t[i], X[i])  # noqa: E731
        else:
            f = lambda i: self.cal(X[i])  # noqa: E731
        Y = np.empty_like(X)
        for i in range(len(X)):
            Y[i] = f(i)
        return correctors(Y, names)


def group_cols(names, members):
    return [i for i, n in enumerate(names) if any(n.startswith(m) for m in members)]


def score(name, X, Y, names, worn, speech):
    ix = {n: i for i, n in enumerate(names)}
    jaw, mc = Y[:, ix["JawOpen"]], Y[:, ix["MouthClosed"]]
    jo = worn & (jaw > 0.15)
    sealed = 100 * np.mean(mc[jo] / jaw[jo] >= 0.9) if jo.any() else float("nan")
    mouth = [i for i, n in enumerate(names) if n.startswith(MOUTH)]
    sp = worn & speech
    pinned_any = 100 * np.mean((Y[sp][:, mouth] >= 0.98).any(axis=1)) if sp.any() else float("nan")
    gains = []
    for i in mouth:
        m = worn & (X[:, i] > 0.02)
        if m.sum() > 50:
            gains.append((np.percentile(Y[m, i] / X[m, i], 99), names[i]))
    g, gname = max(gains)
    quiet = worn & ~speech & (X[:, ix["JawOpen"]] < 0.05)
    qi = np.nonzero(quiet)[0]
    qi = qi[np.r_[False, np.diff(qi) == 1]]  # consecutive pairs only
    jitter = float(np.mean(np.abs(Y[qi][:, mouth] - Y[qi - 1][:, mouth]))) * 1000 if len(qi) else float("nan")
    grp = {gname_: Y[worn][:, group_cols(names, mem)].mean(axis=1) for gname_, mem in GROUPS.items()}
    p95 = {k: float(np.percentile(v, 95)) for k, v in grp.items()}
    p99 = {k: float(np.percentile(v, 99.5)) for k, v in grp.items()}
    return {"name": name, "sealed": sealed, "pinned": pinned_any, "gain": g, "gain_shape": gname,
            "jitter": jitter, "p95": p95, "p99": p99}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--priors", required=True)
    ap.add_argument("--known-good", required=True)
    ap.add_argument("--loops", type=int, default=0, help="also replay everything N times with persistent state")
    ap.add_argument("--spikes", action="store_true", help="also run v0 with injected one-frame spikes")
    a = ap.parse_args(argv)

    recs = [read(p) for p in a.recordings]
    names = recs[0].expressions
    nch = len(recs[0].header["frame"]["channels"])
    n = len(names)
    pri = Priors.load(a.priors).reorder(names)
    kmx, kpr, _, _ = load_calibration(a.known_good, names)
    cands = [
        Candidate("raw", lambda: (lambda x: x)),
        Candidate("stock_known_good", lambda: StockCalibration(n, kmx, kpr, 1.0, True, warm=True)),
        Candidate("stock_reset", lambda: StockCalibration(n, None, None, 1.0, True)),
        Candidate("v0", lambda: AdaptiveCalibrator(pri)),
        Candidate("v0_identity", lambda: AdaptiveCalibrator(Priors.identity(names))),
        Candidate("v0_noguard", lambda: AdaptiveCalibrator(pri, Params(guardrail=False))),
    ]

    data = []
    for r in recs:
        X = r.raw[:, nch:].astype(float)
        data.append((r, r.t, X, worn_mask(r), speech_mask(r, r.t)))

    print(f"# v0 replay evaluation\n\nRecordings: {', '.join(Path(p).name for p in a.recordings)}; "
          f"worn {sum(d[3].sum() for d in data)} frames, speech {sum((d[3] & d[4]).sum() for d in data)} frames.\n")
    results = {}
    for c in cands:
        Ys = [c.run(t, X, names) for (_, t, X, _, _) in data]
        X_all = np.concatenate([d[2] for d in data])
        results[c.name] = score(c.name, X_all, np.concatenate(Ys), names,
                                np.concatenate([d[3] for d in data]), np.concatenate([d[4] for d in data]))
        if c.name.startswith("v0"):
            cal = c.cal
            results[c.name]["c_range"] = (float((cal.c / cal.c0).min()), float((cal.c / cal.c0).max()))

    print("| candidate | lips sealed (% jaw-open frames) | any mouth shape pinned (% speech frames) | "
          "max gain p99 (shape) | rest jitter (x1000) | c / c0 after |")
    print("|---|---|---|---|---|---|")
    for k, s in results.items():
        cr = s.get("c_range")
        print(f"| {k} | {s['sealed']:.1f} | {s['pinned']:.1f} | {s['gain']:.1f}x ({s['gain_shape']}) | "
              f"{s['jitter']:.2f} | {f'{cr[0]:.2f}-{cr[1]:.2f}' if cr else ''} |")

    for key, what in (("p95", "p95 (everyday)"), ("p99", "p99.5 (strongest expressions)")):
        base = results["stock_known_good"][key]
        print(f"\nExpressiveness: {what} of each group, relative to stock_known_good (1.00 = same as now)\n")
        print(f"| group | stock_known_good {key} | " + " | ".join(k for k in results if k != "stock_known_good")
              + " |")
        print("|---|---|" + "---|" * (len(results) - 1))
        for gname in GROUPS:
            row = [f"{results[k][key][gname] / base[gname]:.2f}" if base[gname] > 1e-3 else "-"
                   for k in results if k != "stock_known_good"]
            print(f"| {gname} | {base[gname]:.3f} | " + " | ".join(row) + " |")

    if a.loops:
        print(f"\n## Long run: {a.loops} loops with persistent state\n")
        for c in [x for x in cands if x.name in ("v0", "v0_noguard", "v0_identity")]:
            c.cal = None
            off = 0.0
            for loop in range(a.loops):
                Ys = []
                for (_, t, X, _, _) in data:
                    Ys.append(c.run(t, X, names, persistent=True, t_offset=off))
                    off += float(t[-1]) + 10.0
            cal = c.cal
            s = score(c.name, np.concatenate([d[2] for d in data]), np.concatenate(Ys), names,
                      np.concatenate([d[3] for d in data]), np.concatenate([d[4] for d in data]))
            ratio = cal.c * cal.g / cal.c0
            moved = sorted(((float(ratio[i]), names[i]) for i in range(n) if abs(ratio[i] - 1) > 0.05))
            print(f"- {c.name}: active {cal.active_s / 3600:.2f} h; last loop sealed {s['sealed']:.1f}%, "
                  f"pinned {s['pinned']:.1f}%, max gain {s['gain']:.1f}x ({s['gain_shape']}); "
                  f"c*g/c0 {ratio.min():.2f}-{ratio.max():.2f}; moved >5%: "
                  + ", ".join(f"{nm} {v:.2f}" for v, nm in moved[:6]) + (" ..." if len(moved) > 6 else ""))

    if a.spikes:
        rng = np.random.default_rng(1)
        mouth = [i for i, nm in enumerate(names) if nm.startswith(MOUTH)]
        print("\n## Spike injection (5 random mouth shapes -> 1.0 for one frame, every 5 s)\n")
        for label, inject in (("clean", False), ("spiked", True)):
            cal = AdaptiveCalibrator(pri)
            for (_, t, X, _, _) in data:
                Xs = X.copy()
                if inject:
                    for k in np.searchsorted(t, np.arange(5, t[-1], 5.0)):
                        Xs[k, rng.choice(mouth, 5, replace=False)] = 1.0
                for i in range(len(Xs)):
                    cal.step(t[i], Xs[i])
            print(f"- {label}: c/c0 {(cal.c / cal.c0).min():.3f}-{(cal.c / cal.c0).max():.3f}, "
                  f"peaks recorded {int(cal.peak_n.sum())}")


if __name__ == "__main__":
    main()
