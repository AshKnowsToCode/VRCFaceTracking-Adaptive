"""Replay app runs back to back through adaptive_v0, the way the app ran them, and compare calibrator variants.

Each session (one app run, all its hourly files) starts a fresh calibrator that loads the state the previous one
ended with, as the app does at every start (runtime counters start empty, learned state carries over). The chain
starts from a saved state file of the app (`state-*.json` or a daily snapshot). If that snapshot was written during
the first session (the app saves every 60 active s), `--skip-active S` starts the first session where its active
clock reaches S, so the chain lines up with it.

    python chain_replay.py --state SNAP.json --priors personal-X.json [--skip-active 60] SESSION.vftrec ...
        [--check LATER_SNAP.json ...] [--variant NAME[:key=value,...] ...] [--inject KIND@SECONDS ...]
        [--cache DIR] [--jobs 3] [--report OUT.md]

SESSION.vftrec is any file of a session (the whole session is read). Variants (default: v0-2026-10-13 and current):

  v0-2026-10-13   the calibrator the app ran from 2026-10-11 to 10-18 (no noise cap, old guardrail)
  current         adaptive_v0.Params() as it is now
  NAME:k=v,...    a named variant on top of `current`, e.g. "loose:guard_min_minutes=0"; prefix "v0-2026-10-13+" to
                  start from the old one instead

--check compares the variant named first with a state file the app wrote later (matched by its active seconds):
ceilings, guard factors, rest, noise and held peaks. Use it with the variant that matches the app's build.

--inject adds a synthetic episode to the first session at that session time (seconds), to see what one burst does:
  pucker_burst    60 s of pucker flickering between 0.85 and 1.0 (short pins)
  photo3, photo10 3 or 10 posed open-mouth grins (jaw 0.95, smile 1.0, cheek squint 1.0) of 3 s, 30 s apart
  hold15          a smile held at 0.6 for 15 minutes
  loose           30 minutes of every expression reading 20% high (clipped at 1)

Metrics per session: talking frames (mic > -60 dBFS within 0.15 s), pinned while talking (raw itself, after the
calibrator, made by the calibrator: shown >= 0.98 with raw < 0.9), shown vs raw across the middle of the range for the
everyday shapes, upper-lip left/right balance, largest deadzone, guardrail steps (and how many had no new pins), and
ceilings (c x g) that moved. Expressiveness is judged on the middle of the range: the 2026-10-13 deadzone growth was
invisible to pinning and p95 / p99.5 (long_sessions_2026-10-18.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from adaptive_v0 import AdaptiveCalibrator, Params, Priors  # noqa: E402
from vftrec import read_session  # noqa: E402

LEGACY = dict(sigma_max=1.0, rest_max_rise=1.0, guard_half_life_s=36000.0, guard_needs_new_pins=False,
              guard_min_minutes=0, guard_busy_level=0.0)
MOUTH = ("Jaw", "Mouth", "Lip", "Cheek", "Nose")
SPEECH_DB = -60.0
MID = ["MouthCornerPullRight", "MouthUpperUpRight", "MouthUpperUpLeft", "MouthDimpleLeft", "JawOpen",
       "LipSuckLowerLeft", "LipPuckerUpperRight", "CheekSquintLeft"]
BINS = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.5, 0.6))


# ---------------------------------------------------------------------------------------------------------- input

def load_session(path: str, cache: str | None):
    """(names, t, X float32 raw expressions, audio t, audio dB) of the whole session `path` belongs to."""
    if cache:
        npz = Path(cache) / (Path(path).stem + ".chain.npz")
        if npz.exists():
            d = np.load(npz, allow_pickle=False)
            return list(d["names"]), d["t"], d["x"], d["at"], d["adb"]
    r = read_session(path)
    nch = len(r.header["frame"]["channels"])
    out = (list(r.expressions), r.t, r.raw[:, nch:].astype(np.float32), r.audio_t, r.audio_db)
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez(npz, names=np.array(out[0]), t=out[1], x=out[2], at=out[3], adb=out[4])
    return out


def app_state(path: str) -> tuple[list[str], dict]:
    """An app state file (C# CalibrationState) as adaptive_v0.load_state() input."""
    s = json.load(open(path, encoding="utf-8"))
    return s["Expressions"], {"active_s": s["ActiveSeconds"], "c": s["C"], "g": s["G"], "b": s["B"],
                              "sigma": s["Sigma"], "hist_n": s["HistN"], "hist": s["Hist"], "peak_v": s["PeakV"],
                              "peak_t": s["PeakT"], "peak_n": s["PeakN"]}


def variant_params(spec: str) -> tuple[str, Params]:
    name, _, sets = spec.partition(":")
    base = {}
    if name == "v0-2026-10-13" or name.startswith("v0-2026-10-13+"):
        base = dict(LEGACY)
    defaults = Params()
    for kv in filter(None, sets.split(",")):
        k, _, v = kv.partition("=")
        cur = getattr(defaults, k)  # KeyError-like AttributeError for typos
        base[k] = (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v)
    return name, Params(**base)


def inject(kind: str, at: float, t: np.ndarray, X: np.ndarray, names: list[str]):
    """Adds a synthetic episode in place; returns (start, end) session seconds and the shapes it drives."""
    ix = {n: i for i, n in enumerate(names)}
    cols = lambda prefix: [i for n, i in ix.items() if n.startswith(prefix)]  # noqa: E731
    if kind == "pucker_burst":
        m = (t >= at) & (t < at + 60)
        sq = np.where(((t[m] - at) * 2.5) % 1.0 < 0.5, 1.0, 0.85)
        for i in cols("LipPucker"):
            X[m, i] = sq
        return (at, at + 60), cols("LipPucker")
    if kind in ("photo3", "photo10"):
        k = 3 if kind == "photo3" else 10
        shapes = [ix["JawOpen"]] + cols("MouthCornerPull") + cols("MouthCornerSlant") + cols("CheekSquint")
        for j in range(k):
            m = (t >= at + 30 * j) & (t < at + 30 * j + 3)
            X[np.ix_(m, shapes)] = 1.0
            X[m, ix["JawOpen"]] = 0.95
        return (at, at + 30 * (k - 1) + 3), shapes
    if kind == "hold15":
        m = (t >= at) & (t < at + 900)
        rng = np.random.default_rng(1)
        for i in cols("MouthCornerPull") + cols("MouthCornerSlant"):
            X[m, i] = np.clip(0.6 + rng.normal(0, 0.01, m.sum()), 0, 1)
        return (at, at + 900), cols("MouthCornerPull")
    if kind == "loose":
        m = (t >= at) & (t < at + 1800)
        X[m] = np.minimum(X[m] * 1.2, 1.0)
        return (at, at + 1800), list(range(len(names)))
    raise SystemExit(f"unknown --inject kind {kind!r}")


# ---------------------------------------------------------------------------------------------------------- replay

class Traced(AdaptiveCalibrator):
    """Records every guardrail step with whether new short pins had arrived since the shape's previous step."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.t_now = 0.0
        self.guard_log: list[tuple] = []

    def _update(self, elapsed):
        g0, new0 = self.g.copy(), self._sat_new.copy()
        super()._update(elapsed)
        for i in np.nonzero(self.g > g0 * 1.0001)[0]:
            self.guard_log.append((self.t_now, int(i), float(self.g[i]), bool(new0[i] > 0)))


def run_variant(job: dict) -> dict:
    name, params = variant_params(job["variant"])
    names0, state = app_state(job["state"])
    pri = Priors.load(job["priors"]).reorder(names0)
    checks = sorted((app_state(p)[1] for p in job["checks"]), key=lambda s: s["active_s"]) if job["checks"] else []
    out = {"variant": name, "sessions": [], "checks": []}
    for si, path in enumerate(job["sessions"]):
        names, t, X, at, adb = load_session(path, job["cache"])
        if names != names0:
            raise SystemExit(f"{path}: expression order differs from the state file")
        injected = []
        if si == 0:
            for spec in job["inject"]:
                kind, _, sec = spec.partition("@")
                X = X.copy() if not injected else X
                injected.append((kind, *inject(kind, float(sec), t, X, names)))
        f0 = 0
        if si == 0 and job["skip_active"] > 0:  # gating doesn't depend on the learned state
            probe = AdaptiveCalibrator(pri, params)
            while f0 < len(t) and probe.active_s < job["skip_active"]:
                probe.step(t[f0], X[f0])
                f0 += 1
        cal = Traced(pri, params)
        cal.load_state(json.loads(json.dumps(state)))
        start_cg = cal.c * cal.g
        start_active = cal.active_s
        n = len(t)
        Y = np.zeros((n, len(names)), np.float32)
        gated = np.ones(n, bool)
        dz_max = np.zeros(len(names))
        probes = []  # (session s, active s, c*g) every 5 active minutes, for recovery times
        next_probe = cal.active_s
        for f in range(f0, n):
            cal.t_now = t[f]
            Y[f] = cal.step(t[f], X[f])
            gated[f] = cal.gated
            if cal.active_s >= next_probe:
                next_probe = cal.active_s + 300.0
                dz_max = np.maximum(dz_max, params.deadzone_k * cal.sigma)
                probes.append((float(t[f]), cal.active_s, (cal.c * cal.g).tolist()))
            while checks and cal.active_s >= checks[0]["active_s"] - 1e-6:
                out["checks"].append(compare(cal, checks.pop(0), names, Path(path).name))
        state = cal.state()
        out["sessions"].append(metrics(Path(path).name, names, t, X, Y, gated, at, adb, cal, start_cg, dz_max,
                                       probes, injected, start_active))
        del Y, X
    while checks:
        out["checks"].append(compare(cal, checks.pop(0), names, "end"))
    out["end_state"] = state
    return out


def compare(cal, ref, names, where):
    rcg = np.array(ref["c"]) * np.array(ref["g"])
    rel = np.abs(cal.c * cal.g / rcg - 1)
    peaks = int(np.sum(np.all(np.isclose(np.sort(cal.peak_v, 1), np.sort(np.array(ref["peak_v"]), 1), atol=1e-6), 1)))
    return {"where": where, "active_replay": cal.active_s, "active_app": ref["active_s"],
            "c_max_rel": float(np.max(np.abs(cal.c / np.array(ref["c"]) - 1))), "cg_max_rel": float(rel.max()),
            "cg_worst": names[int(np.argmax(rel))], "b_max_abs": float(np.max(np.abs(cal.b - np.array(ref["b"])))),
            "sigma_max_abs": float(np.max(np.abs(cal.sigma - np.array(ref["sigma"])))), "peaks_equal": peaks}


def metrics(file, names, t, X, Y, gated, at, adb, cal, start_cg, dz_max, probes, injected, start_active):
    worn = ~gated
    loud = np.r_[0, np.cumsum(adb > SPEECH_DB)]
    talk = worn & ((loud[np.searchsorted(at, t + 0.15)] - loud[np.searchsorted(at, t - 0.15)]) > 0)
    mouth = [i for i, n in enumerate(names) if n.startswith(MOUTH)]
    Xs, Ys = X[talk][:, mouth], Y[talk][:, mouth]
    pin_y = Ys >= 0.98
    m = {"file": file, "worn_h": (cal.active_s - start_active) / 3600.0,
         "talk_share": float(talk.sum() / max(1, worn.sum())),
         "pinned_raw": float((Xs >= 0.98).any(axis=1).mean()) if len(Xs) else float("nan"),
         "pinned": float(pin_y.any(axis=1).mean()) if len(Ys) else float("nan"),
         "pinned_made": float((pin_y & (Xs < 0.9)).any(axis=1).mean()) if len(Ys) else float("nan"),
         "made_by_shape": {}, "mid": {}, "deadzone_max": {}, "guard_steps": len(cal.guard_log),
         "guard_steps_without_new_pins": sum(1 for s in cal.guard_log if not s[3]),
         "guard_by_shape": {}, "moved": {}, "injected": []}
    if len(Ys):
        made = (pin_y & (Xs < 0.9)).mean(axis=0)
        for j in np.argsort(-made)[:4]:
            if made[j] > 0:
                m["made_by_shape"][names[mouth[j]]] = float(made[j])
    for nm in MID:
        i = names.index(nm)
        x, y = X[worn, i], Y[worn, i]
        m["mid"][nm] = [float(np.median(y[(x >= lo) & (x < hi)])) if ((x >= lo) & (x < hi)).sum() > 100 else None
                        for lo, hi in BINS]
    for i in np.argsort(-dz_max)[:4]:
        m["deadzone_max"][names[i]] = float(dz_max[i])
    for _, i, g, _new in cal.guard_log:
        m["guard_by_shape"][names[i]] = max(g, m["guard_by_shape"].get(names[i], 0.0))
    end_cg = cal.c * cal.g
    for i in np.nonzero(np.abs(end_cg / start_cg - 1) > 0.05)[0]:
        m["moved"][names[i]] = (float(start_cg[i]), float(end_cg[i]))
    for kind, (a, b), shapes in injected:
        before = next((p for p in reversed(probes) if p[0] < a), probes[0] if probes else None)
        after = next((p for p in probes if p[0] >= b), None)
        if before is None or after is None:
            continue
        base = np.array(before[2])[shapes]
        peak = max((np.array(p[2])[shapes] / base).max() for p in probes if p[0] >= a)
        back = next((p for p in probes if p[0] >= b and (np.array(p[2])[shapes] / base).max() <= 1.02), None)
        m["injected"].append({"kind": kind, "at": a, "max_rise": float(peak),
                              "back_within_2pct_after_active_min": None if back is None else
                              (back[1] - after[1]) / 60.0})
    return m


# ---------------------------------------------------------------------------------------------------------- report

def report(results: list[dict]) -> str:
    L = ["# Chain replay", ""]
    names = [r["variant"] for r in results]
    for r in results:
        for c in r["checks"]:
            L.append(f"Check ({r['variant']} vs app state at {c['active_app']:.1f} active s, {c['where']}): ceilings "
                     f"{c['c_max_rel']:.1e}, c x g {c['cg_max_rel']:.1e} (worst {c['cg_worst']}), rest "
                     f"{c['b_max_abs']:.1e}, noise {c['sigma_max_abs']:.1e}, held peaks equal {c['peaks_equal']} of 88")
    for si, s0 in enumerate(results[0]["sessions"]):
        L += ["", f"## {s0['file']} (worn {s0['worn_h']:.2f} h, talking {s0['talk_share'] * 100:.0f}%)", ""]
        L.append("| | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        row = lambda label, f: L.append(f"| {label} | " + " | ".join(f(r["sessions"][si]) for r in results) + " |")  # noqa: E731
        row("pinned while talking: raw itself", lambda s: f"{s['pinned_raw'] * 100:.2f}%")
        row("pinned while talking: after calibrator", lambda s: f"{s['pinned'] * 100:.2f}%")
        row("pinned while talking: made by calibrator", lambda s: f"{s['pinned_made'] * 100:.2f}%")
        row("  ... by shape", lambda s: ", ".join(f"{k} {v * 100:.2f}%" for k, v in s["made_by_shape"].items()) or "none")
        for nm in MID:
            row(f"{nm} shown at raw 0.1-0.2 / 0.2-0.3 / 0.3-0.4 / 0.5-0.6",
                lambda s, nm=nm: " / ".join("-" if v is None else f"{v:.2f}" for v in s["mid"][nm]))
        row("upper lip L / R at raw 0.2-0.3", lambda s: "-" if None in (s["mid"]["MouthUpperUpLeft"][1],
            s["mid"]["MouthUpperUpRight"][1]) else f"{s['mid']['MouthUpperUpLeft'][1] / max(s['mid']['MouthUpperUpRight'][1], 1e-6):.2f}")
        row("largest deadzones", lambda s: ", ".join(f"{k} {v:.3f}" for k, v in s["deadzone_max"].items()))
        row("guardrail steps (without new pins)", lambda s: f"{s['guard_steps']} ({s['guard_steps_without_new_pins']})"
            + ("; " + ", ".join(f"{k} {v:.2f}" for k, v in s["guard_by_shape"].items()) if s["guard_by_shape"] else ""))
        row("c x g moved > 5%", lambda s: ", ".join(f"{k} {a:.2f}->{b:.2f}" for k, (a, b) in sorted(s["moved"].items()))
            or "none")
        if s0["injected"]:
            row("injected: max rise / back within 2% after (active min)", lambda s: "; ".join(
                f"{e['kind']} {e['max_rise']:.2f}x / " + ("not back" if e["back_within_2pct_after_active_min"] is None
                                                         else f"{e['back_within_2pct_after_active_min']:.0f}")
                for e in s["injected"]))
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", help="one file of each session, in order")
    ap.add_argument("--state", required=True, help="app state file to start from")
    ap.add_argument("--priors", required=True, help="priors the app used (e.g. personal-Virtual_Desktop.json)")
    ap.add_argument("--skip-active", type=float, default=0.0)
    ap.add_argument("--check", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--inject", action="append", default=[])
    ap.add_argument("--cache", help="folder for decoded sessions (large; not on a full drive)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report", help="write the markdown report here too")
    ap.add_argument("--json", help="write the raw results (incl. end states) here")
    a = ap.parse_args(argv)

    variants = a.variant or ["v0-2026-10-13", "current"]
    jobs = [{"variant": v, "state": a.state, "priors": a.priors, "sessions": a.sessions, "cache": a.cache,
             "skip_active": a.skip_active, "checks": a.check if i == 0 else [], "inject": a.inject}
            for i, v in enumerate(variants)]
    if a.cache:  # decode once before the workers start
        for p in a.sessions:
            load_session(p, a.cache)
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(a.jobs, len(jobs))) as ex:
            results = list(ex.map(run_variant, jobs))
    else:
        results = [run_variant(j) for j in jobs]
    text = report(results)
    print(text)
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
