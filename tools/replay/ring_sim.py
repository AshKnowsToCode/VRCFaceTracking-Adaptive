"""Guided-boost ring (DESIGN 8.1, revised 2026-10-11): reference logic for the fill meter, plus a stress test that
replays recorded pose sessions through it with simulated problems.

    python ring_sim.py stress REC.vftrec [REC2 ...] [--report OUT.md]

The ring fills only while the asked-for expression is clearly moving, steady and not overshadowed by an unrelated
one; how fast depends on the level against the person's best so far (the "reference"). The recorded sessions are
open loop (each pose was held ~3 s whatever happened), so the test answers: with the data that hold produced, would
the ring have filled, how long would the person have had to hold, what would it have recorded, and where would it
have stopped to ask (retry or skip)? The person reacting to a tip cannot be replayed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from poses import LATE_S, POSES, REACT_S, Session, find_sessions, load_session  # noqa: E402

HZ = 100.0
FILL_S = 1.5  # full-speed fill time (level >= FULL of the reference)
FULL, HALF = 0.8, 0.5  # full speed at or above FULL, half speed at or above HALF
QUARTER = 0.25  # revised rules: below HALF the ring still fills, at quarter speed, and the pose is flagged
OLD_START = 0.3  # mockup rule: nothing below 0.3 of the reference, and 0.3-0.5 stalls
HELD_S = 0.5  # recorded value = highest level held this long (same as poses.py)
STEADY_WIN_S = 0.5
STEADY_RATIO = 0.7  # steady = over the last 0.5 s, the lowest level is >= 70% of the highest
TREMOR_HOLD_S = 0.25  # revised rules judge steadiness on the level held over one tremor cycle (>= 4 Hz)
COMPANION_MIN = 0.3  # an expression reaching this (of its reference) in the same pose elsewhere moves with it
LIPS_JAW_MIN = 0.35  # lips-pressed pose: jaw at least this open, as a fraction of the wide-open jaw the person just
# did in step 1 (the pose says "about halfway"; nov-ash: 0.18 unusable, 0.54 ok). Works on a weaker headset too.
ASK_AFTER_S = 12.0  # no fill within this much holding -> the ring stops and asks: retry or skip
REF_FOLLOW_AFTER = 3  # revised: after this many finished poses the reference follows the person (see stress())
REF_FOLLOW_MIN = 0.3
# Nothing is out of the Quest Pro (VD) default list (nov-ash, 2026-10-11: Meta supports these channels, so include them;
# the first guided runs confirm). Formerly on hold: these four gave NO channel its highest value in nov-ash's 2026-10-10
# sessions, most likely because the faces didn't match what the tracker calls them:
# press (the deliberate press read 0.00 / 0.00), stretch (the grimace reads as lower-lip down), funnel (the "oh" reads
# as a pucker even with the v2 prompt), jaw_forward (0.00 / 0.30 raw in its pose, 0.64 in the pout). Smile and
# upper_lip stay although everyday use beats their own channels: their leakage sets other ceilings (the smile is the
# strongest stretch, 0.27-0.40 raw vs 0.16 everyday; the upper-lip pose the strongest funnel, 0.92-1.00 vs 0.55-0.68),
# and dropping them would make stretch up to 2.5x and funnel up to 1.7x hotter. Per headset, not global (DESIGN 8.1).
DROPPED: set = set()
UPPER_FACE = ("Eye", "Brow")  # "wrong expression" only compares within a face region (see region())
BOTH_REGIONS = ("NoseSneer", "CheekSquint")  # move with both the eyes and the mouth


def region(name: str) -> set:
    if name.startswith(BOTH_REGIONS):
        return {"upper", "lower"}
    return {"upper"} if name.startswith(UPPER_FACE) else {"lower"}


def mirror(name: str) -> str:
    return name.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")


@dataclass(frozen=True)
class Rules:
    name: str
    own_rest_start: bool  # fill starts when clearly above the person's relaxed face (else: >= 0.3 of the reference)
    steady_on_held: bool  # steadiness on the 0.25 s held level (tremor-tolerant) instead of the raw level
    companions: bool  # expressions known to move with a pose (both sides) don't count as "wrong expression"
    per_side: bool  # two-sided poses fill a left and a right half separately
    regional: bool  # an eye/brow movement never counts against a mouth pose, and the other way round
    dominance: float  # the asked-for level must be >= this x the strongest unrelated expression (both / reference)
    follow_ref: bool  # the reference follows the person after a few poses (a headset that reads everyone weaker)


# mockup = the 2026-10-10 mockup + DESIGN 8.1 as first written (1.5x dominance, no data behind it)
OLD = Rules("mockup", False, False, False, False, False, 1.5, False)
# revised: 1.0x because the sessions show natural co-movement at 0.5-0.9 of the asked-for level (upper lip with a wide
# jaw, brows with wide eyes, mouth with jaw sideways), while the real confusion (an "oh" read as a pucker) is ~2x
NEW = Rules("revised", True, True, True, True, True, 1.0, True)
STEP_COMPANIONS = {"lips_pressed_open": {"JawOpen"}}  # asked for by the pose itself


@dataclass
class Step:
    id: str
    sources: list[str]  # recorded pose ids that feed this guided step
    halves: dict  # "" (whole ring) or "L" / "R" -> shapes
    sideways: bool  # halves fill one after the other (move left, move right), always split

    @property
    def targets(self) -> list[str]:
        return [s for v in self.halves.values() for s in v]


def build_steps() -> list[Step]:
    merged = {"jaw_left": "jaw_sideways", "jaw_right": "jaw_sideways",
              "mouth_left": "mouth_sideways", "mouth_right": "mouth_sideways"}
    sideways = {
        "jaw_sideways": (["jaw_left", "jaw_right"], {"L": ["JawLeft"], "R": ["JawRight"]}),
        "mouth_sideways": (["mouth_left", "mouth_right"],
                           {"L": ["MouthUpperLeft", "MouthLowerLeft"], "R": ["MouthUpperRight", "MouthLowerRight"]}),
    }
    steps, seen = [], set()
    for pid, _, targets in POSES:
        if not targets:
            continue
        if pid in merged:
            sid = merged[pid]
            if sid not in seen:
                seen.add(sid)
                steps.append(Step(sid, *sideways[sid], sideways=True))
            continue
        left = [t for t in targets if t.endswith("Left")]
        right = [t for t in targets if t.endswith("Right")]
        halves = {"L": left, "R": right} if left and right and len(left) + len(right) == len(targets) else {"": targets}
        steps.append(Step(pid, [pid], halves, sideways=False))
    return steps


# ---------------------------------------------------------------------------------------------------- signal helpers


def grid(s: Session, a: float, b: float):
    g = np.arange(a, b, 1.0 / HZ)
    idx = np.clip(np.searchsorted(s.t, g), 1, len(s.t) - 1)
    return g, s.X[idx - 1].copy()  # nearest-previous sample, like poses.py


def trailing(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Trailing rolling fn (np.min / np.max) over w rows; the first rows use what is available."""
    if w <= 1:
        return x.copy()
    pad = np.repeat(x[:1], w - 1, axis=0)
    return fn(np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x]), w, axis=0), axis=-1)


# ------------------------------------------------------------------------------------------------------ scenarios


def _left_frozen(g, G, ctx):
    G = G.copy()
    G[:, ctx["left"]] = ctx["rest_med"][ctx["left"]]
    return G


SCENARIOS = [
    ("as recorded", lambda g, G, ctx: G),
    ("headset reads 50%", lambda g, G, ctx: G * 0.5),
    ("headset reads 30%", lambda g, G, ctx: G * 0.3),
    ("tremor +-30% at 5 Hz", lambda g, G, ctx: np.clip(G * (1 + 0.3 * np.sin(2 * np.pi * 5 * g))[:, None], 0, 1)),
    ("left side not moving", _left_frozen),
]


@dataclass
class Run:
    label: str
    s: Session
    sched: dict
    off: float

    def windows(self) -> dict:
        return {p["id"]: (self.off + p["go"] + REACT_S, self.off + p["end"] + LATE_S) for p in self.sched["poses"]}


def prepare(run: Run, scen_fn):
    """Per recorded pose: (grid times, perturbed data); plus the person's floor (clearly above the relaxed face)."""
    wins = run.windows()
    relax_ids = [pid for pid in ("relax", "relax_end") if pid in wins]
    rest_med = np.median(np.concatenate([grid(run.s, *wins[p])[1] for p in relax_ids]), axis=0)
    ctx = {"left": [i for i, n in enumerate(run.s.names) if n.endswith("Left")], "rest_med": rest_med}
    segs = {}
    for pid, (a, b) in wins.items():
        g, G = grid(run.s, a, b)
        segs[pid] = scen_fn(g, G, ctx)
    relax = np.concatenate([segs[p] for p in relax_ids])
    floor = np.maximum(0.04, 3 * np.percentile(relax, 95, axis=0) + 0.02)  # poses.valid()
    return segs, floor


def companions(step: Step, segs: dict, names: list[str], ref: np.ndarray) -> set:
    G = np.concatenate([segs[p] for p in step.sources])
    held = trailing(G, int(HELD_S * HZ), np.min).max(axis=0)
    tg = set(step.targets)
    comp = {n for j, n in enumerate(names) if n not in tg and held[j] / ref[j] >= COMPANION_MIN}
    return (comp | {mirror(n) for n in comp} | STEP_COMPANIONS.get(step.id, set())) - tg  # both sides


# ---------------------------------------------------------------------------------------------------------- the ring


def run_step(step: Step, segs: dict, names: list[str], ref: np.ndarray, floor: np.ndarray, rules: Rules,
             comp: set, jaw_open_raw: float | None = None, tref: np.ndarray | None = None) -> dict:
    """jaw_open_raw: the held wide-open jaw from the jaw_open step (raw), for the lips-pressed check.
    tref: reference for the asked-for shapes ("your best in this pose"); default = the channel ceilings `ref`.
    Unrelated expressions are always scaled by their channel ceiling."""
    ix = {n: i for i, n in enumerate(names)}
    G = np.concatenate([segs[p] for p in step.sources])
    n, dt = len(G), 1.0 / HZ
    Y = G / ref
    tg = set(step.targets)
    regs = set().union(*(region(s) for s in tg))
    unrel = [j for j, nm in enumerate(names) if nm not in tg and not (rules.companions and nm in comp)
             and (not rules.regional or region(nm) & regs)]
    U = np.where(G >= floor, Y, 0.0)[:, unrel].max(axis=1)  # sensor noise below the floor never counts
    jaw_full = jaw_open_raw if jaw_open_raw else ref[ix["JawOpen"]]
    jaw_ok = G[:, ix["JawOpen"]] >= LIPS_JAW_MIN * jaw_full if step.id == "lips_pressed_open" else np.ones(n, bool)
    halves = step.halves if (rules.per_side or step.sideways) else {"": step.targets}
    held_all = trailing(G, int(HELD_S * HZ), np.min)
    out = {}
    for h, shapes in halves.items():
        cols = [ix[s] for s in shapes]
        lvl = (G[:, cols] / (ref if tref is None else tref)[cols]).mean(axis=1)
        active = (G[:, cols].mean(axis=1) > floor[cols].mean()) if rules.own_rest_start else (lvl >= OLD_START)
        s_ = trailing(lvl, int(TREMOR_HOLD_S * HZ), np.min) if rules.steady_on_held else lvl
        w = int(STEADY_WIN_S * HZ)
        steady = trailing(s_, w, np.min) >= STEADY_RATIO * trailing(s_, w, np.max)
        dominant = lvl >= rules.dominance * U
        low = QUARTER if rules.own_rest_start else 0.0
        rate = np.where(lvl >= FULL, 1.0, np.where(lvl >= HALF, 0.5, low))
        ok = active & jaw_ok & dominant & steady
        inc = np.where(ok, rate, 0.0) * dt / FILL_S
        fill = np.cumsum(inc)
        done = np.flatnonzero(fill >= 1.0)
        end = int(done[0]) + 1 if len(done) else n  # the ring ends the pose when it is full
        lvl_held = trailing(lvl, int(HELD_S * HZ), np.min)
        r = {"filled_s": end * dt if len(done) else None, "need_s": None, "reason": None,
             "weak": bool(rules.own_rest_start and inc[:end][rate[:end] == QUARTER].sum() > 0.5 * inc[:end].sum()),
             "rec_ratio": None,
             "level": float(lvl_held[:end][ok[:end]].max()) if ok[:end].any() else None}  # / the reference used
        if len(done):
            ana = held_all[:, cols].max(axis=0)  # offline analysis: best 0.5 s hold over the whole pose
            okm = ok[:end]
            if okm.any():
                rec = held_all[:end][okm][:, cols].max(axis=0)
                keep = ana > floor[cols]
                if keep.any():
                    r["rec_ratio"] = float(np.mean(rec[keep] / ana[keep]))
        elif fill[-1] > 0:
            r["need_s"] = n * dt / fill[-1]  # same mix of good frames, held longer
        if r["filled_s"] is None and (r["need_s"] is None or r["need_s"] > ASK_AFTER_S):
            if active.mean() < 0.2:
                r["reason"] = "no response"
            else:
                m = active
                counts = {
                    "jaw not open enough": int((m & ~jaw_ok).sum()),
                    "wrong expression": int((m & jaw_ok & ~dominant).sum()),
                    "unsteady": int((m & jaw_ok & dominant & ~steady).sum()),
                    "weaker than target": int((m & jaw_ok & dominant & steady & (rate == 0)).sum()),
                }
                r["reason"] = max(counts, key=counts.get)
        out[h] = r
    return out


def classify(res: dict, step: Step) -> tuple[str, str]:
    """(kind, text): kind = fills (within the recorded hold) / longer (would fill with a longer hold) / asks."""
    parts, kinds = [], []
    for h, r in res.items():
        tag = f"{h} " if h else ""
        if r["filled_s"] is not None:
            kinds.append("fills")
            parts.append(f"{tag}{r['filled_s']:.1f} s" + (" weak" if r["weak"] else ""))
        elif r["reason"] is None:
            kinds.append("longer")
            parts.append(f"{tag}hold ~{r['need_s']:.0f} s" + (" weak" if r["weak"] else ""))
        else:
            kinds.append("asks")
            parts.append(f"{tag}ASK: {r['reason']}")
    kind = "asks" if "asks" in kinds else "longer" if "longer" in kinds else "fills"
    return kind, ", ".join(parts)


# ---------------------------------------------------------------------------------------------------------- stress


def find_runs(paths: list[str]) -> list[Run]:
    scheds = [json.loads(c.read_text(encoding="utf-8")) for c in sorted(HERE.glob("pose_guide_v*.json"))]
    runs = []
    for p in paths:
        s = load_session(p)
        for sc, off, _score, _second in find_sessions(s, scheds):
            runs.append(Run(f"{s.name} @ {off:.0f} s (guide v{sc['version']})", s, sc, off))
    return runs


def stress(paths: list[str], first_ref: str, repeat_ref: str, report: str | None) -> str:
    from adaptive_v0 import Priors

    runs = find_runs(paths)
    if not runs:
        return "No pose session found.\n"
    names = runs[0].s.names
    refs = {"first run": Priors.load(first_ref).reorder(names).c0, "repeat run": Priors.load(repeat_ref).reorder(names).c0}
    steps = build_steps()
    guided = [st for st in steps if st.id not in DROPPED]
    base_full = [prepare(r, SCENARIOS[0][1]) for r in runs]
    base = [b[0] for b in base_full]
    ix = {n: i for i, n in enumerate(names)}

    def other_of(k):
        return (k + 1) % len(runs) if len(runs) > 1 else k

    def comp_for(k, step, ref):  # cross-validated: companions come from the OTHER session when there is one
        return companions(step, base[other_of(k)], names, ref)

    def pose_ref(k, step, ref):
        """Repeat run, revised rules: the reference for the asked-for shapes is "your best in this pose" (the other
        session's best 0.5 s hold in the same pose), so a pose whose own channel the headset reads weakly (upper
        lip on Quest Pro: 0.26 in the pose vs 0.73 in everyday use) isn't judged against the channel's ceiling.
        Falls back to the channel ceiling where that pose never moved the shape."""
        o = other_of(k)
        G = np.concatenate([base[o][p] for p in step.sources])
        best = trailing(G, int(HELD_S * HZ), np.min).max(axis=0)
        t = ref.copy()
        for s_ in step.targets:
            c = ix[s_]
            if best[c] >= max(base_full[o][1][c], 0.25 * ref[c]):
                t[c] = best[c]
        return t

    table, detail = [], {}
    for sname, fn in SCENARIOS:
        prepped = [prepare(r, fn) for r in runs]
        for rules in (OLD, NEW):
            for rname, ref in refs.items():
                agg = {"fills": 0, "longer": 0, "asks": 0}
                weak, ratios, asks, fill_t = 0, [], [], []
                for k, (run, (segs, floor)) in enumerate(zip(runs, prepped)):
                    # reference follows the person: once a few poses are done, a headset that reads everything at
                    # ~40% of the reference stops making every later pose slow. Only downward; it changes the fill
                    # speed and the outline, never the recorded values.
                    scale, levels, jaw_raw = 1.0, [], None
                    for st in steps:
                        tref = pose_ref(k, st, ref) if (rules is NEW and rname == "repeat run") else ref
                        res = run_step(st, segs, names, ref * scale, floor, rules, comp_for(k, st, ref),
                                       jaw_raw if rules.follow_ref else None, tref * scale)
                        if st.id == "jaw_open" and res[""]["level"] is not None:
                            jaw_raw = res[""]["level"] * scale * tref[ix["JawOpen"]]
                        kind, text = classify(res, st)
                        detail[(sname, rules.name, rname, st.id, k)] = text
                        if st.id in DROPPED:
                            continue
                        if rules.follow_ref:
                            levels += [r["level"] * scale for r in res.values() if r["level"] is not None]
                            if len(levels) >= REF_FOLLOW_AFTER:
                                scale = float(np.clip(np.median(levels), REF_FOLLOW_MIN, 1.0))
                        agg[kind] += 1
                        weak += any(r["weak"] for r in res.values())
                        ratios += [r["rec_ratio"] for r in res.values() if r["rec_ratio"] is not None]
                        fill_t += [r["filled_s"] for r in res.values() if r["filled_s"] is not None]
                        if kind == "asks":
                            asks.append(f"{st.id} ({text.split('ASK: ')[-1]})")
                table.append((sname, rules.name, rname, agg, weak, ratios, fill_t, asks))

    total = len(guided) * len(runs)
    L = ["# Guided-boost ring stress test\n",
         f"Sessions: {'; '.join(r.label for r in runs)}. {len(guided)} guided steps per session "
         f"(proposed list: the 27 recorded poses minus relax, with jaw and mouth sideways merged into one step each, "
         f"and without {', '.join(sorted(DROPPED))}), so {total} step runs per row.\n",
         "Open loop: each pose was held ~3 s (window = beep + 0.2 s to end + 0.8 s), whatever happened. "
         "**fills** = the ring would have filled inside that hold; **longer** = it was filling, the person would "
         f"have needed to hold longer (estimate <= {ASK_AFTER_S:.0f} s); **asks** = it would have stopped and asked "
         "(retry or skip). Recorded/analysis = what the ring records when it ends the pose / best 0.5 s hold over "
         "the whole pose (poses.py). References: first run = v0 priors (old stock maxima, stand-in for a typical "
         "Quest Pro value), repeat run = v1 priors (your measured bests); with the revised rules the repeat run also "
         "judges each pose against your best in that pose (taken from the other session). Companions are "
         "cross-validated (taken from the other session).\n",
         "Rules: **mockup** = start at 0.3 of the reference, 0.3-0.5 stalls, raw-level steadiness, the asked-for "
         "expression must be 1.5x any other one anywhere on the face, two-sided poses averaged. **revised** = start "
         "when clearly above your relaxed face, quarter speed below 0.5 of the reference (flagged weak), steadiness "
         "on the 0.25 s held level, known companions allowed (both sides), only the same face region compared, 1.0x, "
         "left and right halves, and after 3 poses the reference follows the person (only downward, at least 0.3).\n",
         "| scenario | rules | reference | fills | longer | asks | flagged weak | fill time median (p90) | "
         "recorded / analysis median (p10) | asks on |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sname, rname, refname, agg, weak, ratios, fill_t, asks in table:
        ft = f"{np.median(fill_t):.1f} s ({np.percentile(fill_t, 90):.1f})" if fill_t else "-"
        rr = f"{np.median(ratios):.2f} ({np.percentile(ratios, 10):.2f})" if ratios else "-"
        L.append(f"| {sname} | {rname} | {refname} | {agg['fills']} | {agg['longer']} | {agg['asks']} | {weak} | "
                 f"{ft} | {rr} | {'; '.join(asks) if asks else '-'} |")
    for rules_name, refname in (("revised", "first run"), ("revised", "repeat run"), ("mockup", "first run")):
        L += ["", f"## Per step: {rules_name} rules, {refname} reference\n",
              "Each cell: session 1 / session 2. L / R = left and right halves.\n",
              "| step | " + " | ".join(s for s, _ in SCENARIOS) + " |",
              "|---" * (len(SCENARIOS) + 1) + "|"]
        for st in steps:
            cells = [" / ".join(detail[(s, rules_name, refname, st.id, k)] for k in range(len(runs)))
                     for s, _ in SCENARIOS]
            label = st.id + (" (dropped)" if st.id in DROPPED else "")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    if report:
        Path(report).write_text(text, encoding="utf-8", newline="\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stress")
    st.add_argument("recordings", nargs="+")
    st.add_argument("--first-ref", default=str(HERE / "priors" / "vd_questpro_novash_v0.json"))
    st.add_argument("--repeat-ref", default=str(HERE / "priors" / "vd_questpro_novash_v1.json"))
    st.add_argument("--report")
    a = ap.parse_args(argv)
    text = stress(a.recordings, a.first_ref, a.repeat_ref, a.report)
    print(text if not a.report else text.split("\n## ")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
