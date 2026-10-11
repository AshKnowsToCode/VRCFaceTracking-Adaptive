"""Pose session: measure a person's full-strength expressions and turn them into calibration priors.

The person plays a spoken audio guide (one pose at a time: prompt, high beep = hold, low beep = relax) while the
fork's Session Recorder is on. No key presses are needed: `analyze` finds the guide in the recording by matching the
whole pose timeline against the data. Manual mode also works: press F8 once per pose, in the listed order.

    python poses.py guide [--out-dir DIR]                      # pose_guide_v1.wav + its timeline (.json)
    python poses.py analyze REC.vftrec [REC2.vftrec ...] --known-good LocalSettings.json \\
        --base-priors priors/vd_questpro_novash_v0.json --out-priors priors/NAME.json --report REPORT.md
    python poses.py selftest                                   # synthetic session -> alignment + measurement

Each REC is one session; two or more sessions also give a repeatability check (full effort varies day to day).
Measured value per shape = the highest level held for >= 0.5 s during its pose(s) (single-frame spikes don't count).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

GUIDE_VERSION = 2  # v2 (2026-10-10): clearer prompts for the lips-pressed and "oh" poses; same poses, same order
SCHEDULE_PATH = HERE / f"pose_guide_v{GUIDE_VERSION}.json"
SR = 22050

HOLD_S = 3.0
REST_S = 1.8
RELAX_HOLD_S = 4.0
GAP_S = 0.4  # prompt end -> high beep
HELD_S = 0.5  # a level counts only if held this long
REACT_S = 0.2  # measuring starts this long after the beep ...
LATE_S = 0.8  # ... and runs this long past the low beep (reaction lag)

LR = ("Left", "Right")


def both(*stems):
    return [s + side for s in stems for side in LR]


# (id, prompt, target shapes). Sideways poses list both directions: which side the module calls "left" is not
# verified, and each shape is measured over every pose that targets it.
POSES = [
    ("relax", "Relax your face completely. Lips together, jaw loose.", []),
    ("jaw_open", "Open your jaw as wide as is comfortable.", ["JawOpen"]),
    ("lips_pressed_open", "Open your jaw about halfway, and keep your lips stretched and pressed together.",
     ["MouthClosed"]),
    ("funnel", "Make a big round O with your lips, open in the middle, like saying oh. Not a kiss.",
     both("LipFunnelUpper", "LipFunnelLower")),
    ("pucker", "Pucker your lips, like a kiss.", both("LipPuckerUpper", "LipPuckerLower")),
    ("lips_in", "Roll both lips into your mouth.", both("LipSuckUpper", "LipSuckLower")),
    ("cheek_puff", "Puff your cheeks full of air.", both("CheekPuff")),
    ("cheek_suck", "Suck your cheeks in. Fish face.", both("CheekSuck")),
    ("smile", "Big smile. Show your teeth.", both("MouthCornerPull", "MouthCornerSlant", "CheekSquint")),
    ("upper_lip", "Raise your upper lip to show your top teeth.", both("MouthUpperUp", "MouthUpperDeepen")),
    ("lower_lip", "Pull your lower lip down to show your bottom teeth.", both("MouthLowerDown")),
    ("frown", "Pull the corners of your mouth down. A sad frown.", both("MouthFrown")),
    ("stretch", "Stretch your mouth wide and flat, like a grimace.", both("MouthStretch")),
    ("press", "Press your lips together hard.", both("MouthPress", "MouthTightener")),
    ("chin_raise", "Push your lower lip up and out. A pout.", ["MouthRaiserLower", "MouthRaiserUpper"]),
    ("sneer", "Wrinkle your nose. A disgusted sneer.", both("NoseSneer")),
    ("jaw_left", "Move your jaw to the left.", ["JawLeft", "JawRight"]),
    ("jaw_right", "Move your jaw to the right.", ["JawLeft", "JawRight"]),
    ("jaw_forward", "Push your jaw forward.", ["JawForward"]),
    ("mouth_left", "Lips together. Shift your whole mouth to the left.", both("MouthUpper", "MouthLower")),
    ("mouth_right", "Lips together. Shift your whole mouth to the right.", both("MouthUpper", "MouthLower")),
    ("squint", "Squint your eyes hard, but keep them open.", both("EyeSquint")),
    ("eyes_wide", "Open your eyes as wide as you can.", both("EyeWide")),
    ("brows_up", "Raise your eyebrows as high as you can.", both("BrowInnerUp", "BrowOuterUp")),
    ("brows_down", "Pull your eyebrows down and together. Angry.", both("BrowLowerer", "BrowPinch")),
    ("tongue_out", "Stick your tongue out.", ["TongueOut"]),
    ("relax_end", "Relax. That was the last one. Thank you.", []),
]
INTRO = ("Pose session. Do each face at full strength, but comfortably. Start at the high beep, hold it steady, "
         "and relax at the low beep. Here we go.")

# What on the avatar reads each input (nov-ash's avatars, CustomFT controller + VRCFT settings as of 2026-10-09).
# Used to say where a change in boost will show and which hand-tuned numbers were set against the old values.
AVATAR_IMPACT = [
    (("CheekPuff", "CheekSuck"), "v2/CheekPuffSuck L/R -> Cheek Puff trees: dead zone 0.3 (09-07, set against the old "
                                 "~8x puff boost), 3-bit synced"),
    (("MouthCornerPull", "MouthCornerSlant"), "v2/SmileFrown L/R -> smile hooding (@0.5, open-squint @1.5), pupil "
                                              "warmth (dilate @2), emulated cheek squint (@4)"),
    (("MouthFrown",), "v2/SmileFrown L/R (negative) -> pupil sad-dilate (@-2), mouth corner trees"),
    (("CheekSquint",), "v2/CheekSquint L/R -> Cheek Raise trees (CheekSquint 40 / lower-lid squint fade)"),
    (("EyeSquint",), "VRCFT Parameter Adjustment eyeSquint ceiling 0.63 on top; v2/EyeSquint -> cheek-raise fade, "
                     "pupil squint (@4)"),
    (("EyeWide",), "VRCFT Parameter Adjustment eyeWide ceiling 0.75 on top; v2/EyeLid -> lid trees (wide over "
                   "0.75-0.85), pupil shock (0.78-0.85)"),
    (("BrowInnerUp", "BrowOuterUp"), "VRCFT Parameter Adjustment eyeBrowRaiser ceiling 0.59 on top; "
                                     "v2/BrowExpression L/R -> brow trees"),
    (("BrowLowerer", "BrowPinch"), "VRCFT Parameter Adjustment eyeBrowLower ceiling 0.46 on top; angry brows "
                                   "softened to 35% (09-04), Brow Hood full @0.6 (09-15), furrow damp 40%, "
                                   "pupil glare (@2)"),
    (("JawOpen",), "v2/JawOpen -> Jaw Open Blend (FaceTrackingLimits); adds to visemes while both are on"),
    (("MouthClosed",), "v2/MouthClosed -> lips-together shape; VRCFT side: scaled with JawOpen (coupling ratio)"),
    (("LipPucker",), "v2/LipPucker -> custom Lip_Pucker_custom clip (strength set by eye)"),
    (("LipFunnel",), "v2/LipFunnel -> custom Lip_Funnel clip (strength set by eye)"),
    (("MouthPress",), "v2/MouthPress -> custom Mouth_Press clip (strength set by eye)"),
    (("NoseSneer",), "v2/NoseSneer -> custom Nose_Sneer clip (strength set by eye)"),
    (("MouthUpper", "MouthLower"), "v2/MouthX -> mouth sideways (template)"),
    (("LipSuck",), "v2/LipSuckUpper/Lower (template); VRCFT LipSuck limiter"),
]


def impact_for(shape: str) -> str:
    for stems, text in AVATAR_IMPACT:
        if any(shape.startswith(s) for s in stems) and not (
                shape.startswith(("MouthUpperUp", "MouthUpperDeepen", "MouthLowerDown")) and "MouthUpper" in stems):
            return text
    return "template trees (no hand-tuned numbers)"


# ---------------------------------------------------------------------------------------------------- audio guide


def _tts(texts: list[str], workdir: Path, voice: str) -> list[np.ndarray]:
    items = [{"file": str(workdir / f"p{i:02d}.wav"), "text": t} for i, t in enumerate(texts)]
    lpath = workdir / "prompts.txt"
    lpath.write_text("".join(f"{it['file']}\t{it['text']}\n" for it in items), encoding="utf-16")
    # Windows Script Host + SAPI COM; PowerShell was dropped because it hung at startup on this PC (2026-10-10)
    subprocess.run(["cscript", "//nologo", str(HERE / "tts_sapi.vbs"), str(lpath), voice], check=True, timeout=300)
    out = []
    for it in items:
        with wave.open(it["file"]) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            a = np.frombuffer(w.readframes(n), dtype=np.int16 if sw == 2 else np.uint8).astype(float)
        if sw == 1:
            a = (a - 128) * 256
        a = a.reshape(-1, ch).mean(axis=1) / 32768.0
        if sr != SR:
            a = np.interp(np.arange(0, len(a) * SR / sr) * sr / SR, np.arange(len(a)), a)
        loud = np.nonzero(np.abs(a) > 0.02 * np.abs(a).max())[0]
        a = a[max(loud[0] - int(0.03 * SR), 0): loud[-1] + int(0.08 * SR)]
        out.append(0.7 * a / np.abs(a).max())
    return out


def _tone(freq: float, dur: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1.0, np.minimum(t, dur - t) / 0.01)
    return 0.35 * env * np.sin(2 * math.pi * freq * t)


def build_guide(out_dir: Path, voice: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        speech = _tts([INTRO] + [p[1] for p in POSES], Path(td), voice)
    hi, lo = _tone(880, 0.15), _tone(523, 0.25)
    chunks, t, sched = [], 0.0, []

    def put(a):
        nonlocal t
        chunks.append(a)
        t += len(a) / SR

    def silence(s):
        put(np.zeros(int(s * SR)))

    silence(1.0)
    put(speech[0])
    silence(2.0)
    for (pid, _, targets), clip in zip(POSES, speech[1:]):
        p0 = t
        put(clip)
        silence(GAP_S)
        go = t
        put(hi)
        silence((RELAX_HOLD_S if not targets else HOLD_S) - len(hi) / SR)
        end = t
        put(lo)
        silence(REST_S)
        sched.append({"id": pid, "prompt": round(p0, 3), "go": round(go, 3), "end": round(end, 3),
                      "targets": targets})
    silence(1.0)
    audio = np.concatenate(chunks)
    wav = out_dir / f"pose_guide_v{GUIDE_VERSION}.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    js = {"version": GUIDE_VERSION, "duration": round(t, 3), "hold_s": HOLD_S, "poses": sched}
    jpath = out_dir / f"pose_guide_v{GUIDE_VERSION}.json"
    jpath.write_text(json.dumps(js, indent=1) + "\n", encoding="utf-8")
    SCHEDULE_PATH.write_text(json.dumps(js, indent=1) + "\n", encoding="utf-8")
    return wav, jpath


# ---------------------------------------------------------------------------------------------------- analysis


@dataclass
class Session:
    name: str
    t: np.ndarray  # (n,) seconds
    X: np.ndarray  # (n, k) raw expression weights (module output)
    names: list[str]
    markers: list[float]
    header: dict | None = None


def load_session(path: str) -> Session:
    from vftrec import read

    r = read(path)
    nch = len(r.header["frame"]["channels"])
    marks = [float(s) for s in r.seconds([m[0] for m in r.markers])] if r.markers else []
    return Session(Path(path).name, r.t, r.raw[:, nch:].astype(float), r.expressions, marks, r.header)


def header_calibration(header: dict | None, names: list[str]):
    """The stock calibration maxima in effect when the recording started (None if absent / empty)."""
    if not header:
        return None
    for m in header.get("mutations", []):
        if m.get("name") == "Calibration" and m.get("active"):
            shapes = {s["name"]: s["max"] for s in m.get("state", {}).get("calData", {}).get("Shapes", [])}
            if shapes and any(v > 0 for v in shapes.values()):
                return {n: float(shapes.get(n, 0.0)) for n in names}
    return None


def _grid(s: Session, hz=100.0):
    g = np.arange(s.t[0], s.t[-1], 1.0 / hz)
    idx = np.clip(np.searchsorted(s.t, g), 1, len(s.t) - 1)
    # nearest-previous sample (no smoothing across frames)
    G = s.X[idx - 1]
    return g, G


def _held(seg: np.ndarray, hz: float) -> np.ndarray:
    """Highest level held for >= HELD_S in each column of seg (rows = time on the grid)."""
    w = max(int(round(HELD_S * hz)), 1)
    if len(seg) < w:
        return np.zeros(seg.shape[1])
    win = np.lib.stride_tricks.sliding_window_view(seg, w, axis=0)  # (n-w+1, k, w)
    return win.min(axis=-1).max(axis=0)


def _rolling_min(G: np.ndarray, w: int) -> np.ndarray:
    """Rolling minimum over w rows, streaming (no (n, k, w) temporary)."""
    if len(G) < w:
        return np.zeros((0, G.shape[1]))
    out = G[: len(G) - w + 1].copy()
    for k in range(1, w):
        np.minimum(out, G[k: len(G) - w + 1 + k], out=out)
    return out


def everyday_held(paths: list[str], names: list[str], q=99.95, hz=50.0) -> dict:
    """Per shape: a high quantile of the 0.5 s held level over normal (worn) use. What the face really does."""
    from priors import worn_mask
    from vftrec import read

    parts = []
    for p in paths:
        r = read(p)
        nch = len(r.header["frame"]["channels"])
        X = r.raw[:, nch:].astype(float)
        X = np.where(worn_mask(r)[:, None], X, 0.0)  # headset-off frames cannot form a held level
        s = Session(Path(p).name, r.t, X, r.expressions, [])
        _, G = _grid(s, hz)
        parts.append(_rolling_min(G, int(round(HELD_S * hz))))
    H = np.concatenate(parts)
    return dict(zip(names, np.percentile(H, q, axis=0).tolist()))


def score_curve(s: Session, sched: dict, hz=50.0):
    """Match score of the guide timeline at every candidate start time (recording seconds)."""
    ix = {n: i for i, n in enumerate(s.names)}
    g = np.arange(s.t[0], s.t[-1], 1.0 / hz)
    idx = np.clip(np.searchsorted(s.t, g), 1, len(s.t) - 1)
    G = s.X[idx - 1]
    # per-shape scale; the 0.1 floor keeps rarely used shapes from inflating the score in long idle recordings
    scale = np.maximum(np.percentile(G, 99.9, axis=0), 0.1)
    S = G / scale
    poses = [p for p in sched["poses"] if p["targets"]]
    cums = []
    for p in poses:
        cols = [ix[n] for n in p["targets"] if n in ix]
        a = S[:, cols].mean(axis=1) if cols else np.zeros(len(g))
        cums.append(np.r_[0.0, np.cumsum(a)])
    n = len(g)
    offs = np.arange(-sched["poses"][0]["prompt"], g[-1] - g[0] - sched["duration"] + 5.0, 1.0 / hz)
    if len(offs) < 1:
        return np.zeros(0), np.zeros(0)

    def mean_win(c, a, b):  # window [a, b) in grid samples, vectorised over offsets
        a = np.clip(a, 0, n)
        b = np.clip(b, 0, n)
        return np.where(b > a, (c[b] - c[a]) / np.maximum(b - a, 1), 0.0)

    score = np.zeros(len(offs))
    base = (offs * hz).astype(int)
    for p, c in zip(poses, cums):
        h0, h1 = base + int((p["go"] + 0.5) * hz), base + int((p["end"]) * hz)
        r0, r1 = base + int((p["end"] + 0.6) * hz), base + int((p["end"] + REST_S - 0.1) * hz)
        q0, q1 = base + int((p["prompt"] - REST_S + 0.2) * hz), base + int(p["prompt"] * hz)
        score += mean_win(c, h0, h1) - 0.5 * (mean_win(c, r0, r1) + mean_win(c, q0, q1))
    score /= len(poses)
    return g[0] + offs, score


def align(s: Session, sched: dict, hz=50.0):
    """Best start time (recording seconds) of one guide version, its score, and the best score elsewhere."""
    t0, score = score_curve(s, sched, hz)
    if not len(score):
        return None, 0.0, 0.0
    k = int(np.argmax(score))
    away = np.abs(t0 - t0[k]) > 3.0
    second = score[away].max() if away.any() else 0.0
    return float(t0[k]), float(score[k]), float(second)


MATCH_MIN = 0.35  # a real session scores ~0.6-0.9; normal use without a session stays below ~0.2


def find_sessions(s: Session, scheds: list[dict]):
    """Every guide run in one recording (several sessions, any guide versions): greedy best match, mask its span
    for all versions, repeat. Returns [(sched, start, score, best score elsewhere)] in time order."""
    curves = [(sc, *score_curve(s, sc)) for sc in scheds]
    taken, found = [], []
    while True:
        best = None
        for sc, t0, score in curves:
            if not len(score):
                continue
            ok = np.ones(len(t0), bool)
            for a, b in taken:
                ok &= (t0 < a - sc["duration"] - 3.0) | (t0 > b + 3.0)
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, score, -np.inf)))
            if best is None or score[k] > best[2]:
                away = ok & (np.abs(t0 - t0[k]) > 3.0)
                best = (sc, float(t0[k]), float(score[k]), float(score[away].max()) if away.any() else 0.0)
        if best is None or best[2] < MATCH_MIN:
            break
        found.append(best)
        taken.append((best[1], best[1] + best[0]["duration"]))
    return sorted(found, key=lambda f: f[1])


@dataclass
class PoseResult:
    held: dict  # pose id -> held vector (all shapes) during that pose's window
    rest_p95: dict  # shape -> 95th percentile during the relax poses
    ratio: tuple | None  # (MouthClosed / JawOpen during the lips-pressed pose, jaw level)


def measure(s: Session, sched: dict, offset: float | None, hz=100.0) -> PoseResult:
    g, G = _grid(s, hz)
    ix = {n: i for i, n in enumerate(s.names)}
    poses = sched["poses"]
    if offset is None:  # manual mode: one marker per pose
        wins = []
        for k, m in enumerate(s.markers[:len(poses)]):
            nxt = s.markers[k + 1] if k + 1 < len(s.markers) else m + 8.0
            wins.append((m + 0.3, min(m + 7.0, nxt)))
    else:
        wins = [(offset + p["go"] + REACT_S, offset + p["end"] + LATE_S) for p in poses]
    held, rest = {}, []
    for p, (a, b) in zip(poses, wins):
        m = (g >= a) & (g < b)
        if m.sum() < HELD_S * hz:
            continue
        held[p["id"]] = _held(G[m], hz)
        if not p["targets"]:
            rest.append(G[m])
    rest = np.concatenate(rest) if rest else s.X
    rest_p95 = {n: float(np.percentile(rest[:, i], 95)) for n, i in ix.items()}
    ratio = None
    lp = [k for k, p in enumerate(poses) if p["id"] == "lips_pressed_open"]
    if lp and lp[0] < len(wins):
        a, b = wins[lp[0]]
        m = (g >= a) & (g < b)
        jaw, mc = G[m][:, ix["JawOpen"]], G[m][:, ix["MouthClosed"]]
        open_ = jaw > 0.1
        if open_.sum() > HELD_S * hz:
            ratio = (float(np.median(mc[open_] / jaw[open_])), float(np.median(jaw[open_])))
    return PoseResult(held, rest_p95, ratio)


def valid(shape: str, value: float, rest_p95: dict) -> bool:
    return value >= max(0.04, 3 * rest_p95.get(shape, 0.0) + 0.02)


LEAK_FRACTION = 0.8  # deliberate pose below this fraction of the strongest other source = leakage-dominated
SIDEWAYS = [("JawLeft", "JawRight"), ("MouthUpperLeft", "MouthUpperRight"), ("MouthLowerLeft", "MouthLowerRight")]
COUPLING_MIN_JAW = 0.3  # the lips-pressed pose only sets the coupling ratio if the jaw was this open


def evidence(shape: str, sched: dict, pr: PoseResult, names: list[str], everyday: dict) -> dict:
    """Where a shape's ceiling evidence comes from in one session."""
    i = names.index(shape)
    tgt_poses = [p["id"] for p in sched["poses"] if shape in p["targets"] and p["id"] in pr.held]
    target = max((pr.held[pid][i] for pid in tgt_poses), default=0.0)
    others = [(float(v[i]), pid) for pid, v in pr.held.items() if pid not in tgt_poses]
    other_v, other_p = max(others, default=(0.0, "-"))
    ev = float(everyday.get(shape, 0.0))
    if not valid(shape, target, pr.rest_p95):
        basis = "not seen"
    elif target < LEAK_FRACTION * other_v and other_v >= ev:
        basis = "leak"  # another pose drives this channel harder than its own pose
    elif target < LEAK_FRACTION * ev:
        basis = "everyday"  # normal use goes further than the pose did: the pose under-performed
    else:
        basis = "measured"
    return {"target": float(target), "other": other_v, "other_pose": other_p, "everyday": ev, "basis": basis}


def choose(ev: dict, now: float) -> float:
    """Ceiling for one session. Never below anything the shape actually reached (no pinning on real expressions);
    a shape the deliberate pose does not drive is never made hotter than now."""
    seen = max(ev["target"], ev["other"], ev["everyday"])
    if ev["basis"] == "measured":
        return seen
    return max(now, seen) if now > 0 else max(seen, 1.0 if ev["basis"] == "not seen" else seen)


def analyze(paths: list[str], schedule: Path, known_good: str | None, base_priors: str | None,
            out_priors: str | None, report: str | None, everyday_paths: list[str] | None = None) -> str:
    from adaptive_v0 import Params, Priors
    from stock import load_calibration

    # every guide version has the same poses in the same order, only the timing differs: each session is matched
    # against every known version (or the one given) and measured with the version that fits
    cands = [Path(schedule)] if schedule else sorted(HERE.glob("pose_guide_v*.json"))
    scheds = [json.loads(c.read_text(encoding="utf-8")) for c in cands]
    sched = scheds[-1]
    sessions = [load_session(p) for p in paths]
    names = sessions[0].names
    p_ = Params()
    lines = ["# Pose session analysis\n", f"Guide versions tried: {', '.join('v' + str(x['version']) for x in scheds)}, "
             f"{len(sched['poses'])} poses. Sessions: {', '.join(s.name for s in sessions)}."]
    results = []
    for s in sessions:
        if len(s.markers) >= len(sched["poses"]):
            lines.append(f"- {s.name}: manual (one F8 marker per pose).")
            results.append((s, measure(s, sched, None)))
            continue
        found = find_sessions(s, scheds)
        if not found:
            curves = [score_curve(s, sc)[1] for sc in scheds]
            best = max((float(c.max()) for c in curves if len(c)), default=0.0)
            lines.append(f"- {s.name}: NO pose session found (best match {best:.2f}, needs {MATCH_MIN}). Skipped.")
            continue
        for use, off, best, second in found:
            label = f"{s.name} @ {off:.0f} s (guide v{use['version']})" if len(found) > 1 else s.name
            lines.append(f"- {label}: guide v{use['version']} found at {off:.1f} s (match {best:.2f}).")
            results.append((replace(s, name=label), measure(s, use, off)))
    if not results:
        text = "\n".join(lines) + "\n"
        if report:
            Path(report).write_text(text, encoding="utf-8")
        return text

    # current values: an explicit LocalSettings.json, else the calibration in effect when the first session started
    if known_good:
        mx, _, _, _ = load_calibration(known_good, names)
        now, now_src = dict(zip(names, mx.tolist())), Path(known_good).name
    else:
        now = header_calibration(results[0][0].header, names)
        now_src = "the calibration in effect at the start of " + results[0][0].name
        if now is None:
            base0 = Priors.load(base_priors).reorder(names) if base_priors else Priors.identity(names)
            now, now_src = dict(zip(names, base0.c0.tolist())), "the base priors"
    everyday = everyday_held(everyday_paths, names) if everyday_paths else {}
    lines.append(f"- Current values: {now_src}. Everyday use: "
                 + (", ".join(Path(p).name for p in everyday_paths) if everyday_paths else "none given") + ".\n")

    targeted = [n for ps in sched["poses"] for n in ps["targets"]]
    targeted = list(dict.fromkeys(n for n in targeted if n in names))
    per_shape = {}
    for n in targeted:
        evs = [evidence(n, sched, pr, names, everyday) for _, pr in results]
        vals = [choose(e, now.get(n, 0.0)) for e in evs]
        measured = [v for v, e in zip(vals, evs) if e["basis"] == "measured"]
        if measured and len(measured) * 2 >= len(evs):
            c = float(np.exp(np.mean(np.log(measured))))
            basis = "measured"
        else:
            c = max(vals)
            # conservative value (max) but the most informative label across sessions
            basis = min((e["basis"] for e in evs), key=["measured", "everyday", "leak", "not seen"].index)
        spread = (max(measured) / min(measured)) if len(measured) > 1 else None
        per_shape[n] = {"c": c, "basis": basis, "ev": evs[0], "evs": evs, "spread": spread, "n": len(measured)}

    # sideways: the module's left/right is unreliable on this headset, so both sides share the larger ceiling
    sideways_note = []
    for a, b in SIDEWAYS:
        if a in per_shape and b in per_shape:
            hi = max(per_shape[a]["c"], per_shape[b]["c"])
            for n in (a, b):
                per_shape[n]["c"] = hi
                per_shape[n]["basis"] += ", L/R pooled"
    for pa, pb, la, lb in (("jaw_left", "jaw_right", "JawLeft", "JawRight"),
                           ("mouth_left", "mouth_right", "MouthUpperLeft", "MouthUpperRight")):
        reads, opposite = [], False
        for s_, pr in results:
            if pa in pr.held and pb in pr.held:
                ia, ib = names.index(la), names.index(lb)
                fmt_ = lambda pid: f"{la} {pr.held[pid][ia]:.2f} / {lb} {pr.held[pid][ib]:.2f}"  # noqa: E731
                same = (pr.held[pa][ia] > pr.held[pa][ib]) == (pr.held[pb][ia] > pr.held[pb][ib])
                opposite |= not same
                reads.append(f"{s_.name}: {pa} -> {fmt_(pa)}, {pb} -> {fmt_(pb)}"
                             + (" (same side both times)" if same else " (opposite sides)"))
        if reads:
            verdict = ("both directions are sensed (a session read opposite sides); a session that read the same side "
                       "twice most likely moved the same way twice. Which label is which way can't be told from the "
                       "data: check the avatar in a mirror" if opposite else
                       "only one side ever responded - either the headset labels both directions the same, or both "
                       "moves went the same way; check in a mirror")
            sideways_note.append("; ".join(reads) + ". So: " + verdict)

    s0, pr0 = results[0]
    lines.append("## Detection (first session)\n")
    lines.append("| pose | targets held (>= 0.5 s) | strongest other shapes (cross-talk) |")
    lines.append("|---|---|---|")
    ix = {n: i for i, n in enumerate(names)}
    for ps in sched["poses"]:
        if not ps["targets"] or ps["id"] not in pr0.held:
            continue
        hv = pr0.held[ps["id"]]
        tv = ", ".join(f"{n} {hv[ix[n]]:.2f}" + ("" if valid(n, hv[ix[n]], pr0.rest_p95) else " (not seen)")
                       for n in ps["targets"] if n in ix)
        tgt = {ix[n] for n in ps["targets"] if n in ix}
        xs = sorted(((hv[i], names[i]) for i in range(len(names)) if i not in tgt and hv[i] > 0.15), reverse=True)
        lines.append(f"| {ps['id']} | {tv} | {', '.join(f'{n} {v:.2f}' for v, n in xs[:4]) or '-'} |")

    lines.append("\n## Findings\n")
    leak = [n for n, d in per_shape.items() if d["basis"].startswith("leak")]
    weak = [n for n, d in per_shape.items() if d["basis"].startswith("everyday")]
    unseen = [n for n, d in per_shape.items() if d["basis"].startswith("not seen")]
    if unseen:
        lines.append(f"- **Not sensed in their pose** (kept at current values): {', '.join(unseen)}.")
    if leak:
        lines.append("- **Driven by other expressions, not by their own pose** (never made hotter than now): "
                     + "; ".join(f"{n} (pose {per_shape[n]['ev']['target']:.2f}, "
                                 f"{per_shape[n]['ev']['other_pose']} {per_shape[n]['ev']['other']:.2f}, "
                                 f"everyday {per_shape[n]['ev']['everyday']:.2f})" for n in leak) + ".")
    if weak:
        lines.append("- **Everyday use went further than the pose** (ceiling from everyday use, never hotter than now): "
                     + "; ".join(f"{n} (pose {per_shape[n]['ev']['target']:.2f}, everyday "
                                 f"{per_shape[n]['ev']['everyday']:.2f})" for n in weak) + ".")
    for note in sideways_note:
        lines.append(f"- **Sideways direction:** {note}. Both sides share one ceiling (the larger), so a direction "
                     "label mix-up can't make one side hotter.")

    lines.append("\n## Ceilings\n")
    lines.append("Ceiling = the raw value that maps to 100%; boost = 1 / ceiling (capped at "
                 f"{1 / p_.c_abs_min:.1f}x in v0; the stock calibrator has no cap). Change > 1 = hotter than now. A ceiling is never set below what the shape "
                 "reached in any pose or in everyday use (held >= 0.5 s), so real expressions don't pin.\n")
    lines.append("| shape | now | own pose | strongest elsewhere | everyday | new | boost now -> new | change | basis | "
                 "what on the avatar reads it |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, d in per_shape.items():
        evs, c, nw = d["evs"], d["c"], now.get(n, 0.0)
        bn = 1.0 / nw if nw > 0 else 1.0  # stock has no cap
        bnew = 1.0 / max(c, p_.c_abs_min)
        sp = f" ({d['n']} sessions, spread {d['spread']:.2f}x)" if d["spread"] else ""
        own = " / ".join(f"{e['target']:.2f}" for e in evs)
        top = max(evs, key=lambda e: e["other"])
        lines.append(f"| {n} | {nw:.3f} | {own} | {top['other_pose']} {top['other']:.2f} | "
                     f"{evs[0]['everyday']:.2f} | {c:.3f} | {bn:.2f}x -> {bnew:.2f}x | {bnew / bn:.2f} | "
                     f"{d['basis']}{sp} | {impact_for(n)} |")

    lines.append("\n## Lips pressed with the jaw open\n")
    cr = None
    for (s_, pr) in results:
        if pr.ratio:
            r, jaw = pr.ratio
            lines.append(f"- {s_.name}: headset MouthClosed / JawOpen {r:.2f} with the jaw at {jaw:.2f}"
                         + ("" if jaw >= COUPLING_MIN_JAW else f" (jaw under {COUPLING_MIN_JAW}: not used)") + ".")
    good = [pr.ratio for _, pr in results if pr.ratio and pr.ratio[1] >= COUPLING_MIN_JAW]
    if good:
        r = float(np.median([x[0] for x in good]))
        cr = min(max(1.0 / max(r, 1e-6), 1.0), 2.0)
        lines.append(f"\nCoupling ratio that makes the deliberate pose exactly sealed: **{cr:.2f}** (from "
                     f"{len(good)} usable pose(s)): the avatar's lips seal once the headset reports lips-pressed at "
                     f"{100 / cr:.0f}% of jaw-open or more.")
    else:
        lines.append("\nNo usable pose (jaw never at 0.3 or more): kept the current coupling ratio.")

    lr = []
    for n, d in per_shape.items():
        m = n[:-4] + "Right" if n.endswith("Left") else None
        if m in per_shape and not any(n in pair for pair in SIDEWAYS):
            q = d["c"] / per_shape[m]["c"]
            if abs(math.log(q)) > math.log(1.15):
                lr.append(f"{n[:-4]} L {d['c']:.2f} / R {per_shape[m]['c']:.2f}")
    lines.append("\n## Left / right differences over 15%\n")
    lines.append(("- " + "\n- ".join(lr)) if lr else "None.")

    lines.append("\n## Avatar settings to re-check if these ceilings go live (boost change over 25%)\n")
    seen_txt = {}
    for n, d in per_shape.items():
        nw = now.get(n, 0.0)
        bn = 1.0 / nw if nw > 0 else 1.0  # stock has no cap
        ch = (1.0 / max(d["c"], p_.c_abs_min)) / bn
        if abs(math.log(ch)) > math.log(1.25):
            seen_txt.setdefault(impact_for(n), []).append(f"{n} {ch:.2f}x")
    for txt, items in seen_txt.items():
        lines.append(f"- {', '.join(items)}: {txt}")

    if out_priors:
        base = Priors.load(base_priors).reorder(names) if base_priors else Priors.identity(names)
        c0 = base.c0.copy()
        for i, n in enumerate(names):
            if n in per_shape:
                c0[i] = per_shape[n]["c"]
        coupled = dict(base.coupled)
        if cr is not None:
            coupled["MouthClosed"] = {"parent": "JawOpen", "ratio": round(cr, 4)}
        meta = dict(base.meta)
        meta.update({"name": Path(out_priors).stem,
                     "c0_source": "pose sessions " + ", ".join(s.name for s, _ in results)
                                  + ("; everyday " + ", ".join(Path(p).name for p in everyday_paths)
                                     if everyday_paths else ""),
                     "c0_basis": {n: d["basis"] for n, d in per_shape.items()},
                     "c0_fallback": base.meta.get("name", "identity"),
                     "note": "Ceilings from full-strength pose holds (>= 0.5 s), never below any held level seen in "
                             "the poses or everyday use; leakage-driven shapes never hotter than the values in "
                             "effect. Unposed shapes keep the base priors."})
        Priors(names, c0, base.b0, base.sigma0, coupled, meta).save(out_priors)
        lines.append(f"\nWrote priors: `{out_priors}`.")
    text = "\n".join(lines) + "\n"
    if report:
        Path(report).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------------------------------- self-test


def synthetic_session(seed: int, true_off: float, effort: float = 1.0):
    """A fake recording of the guide: known offset and maxima, reaction lag, cross-talk, noise, one-frame spikes,
    talking before the session. Returns (Session, truth dict)."""
    from adaptive_v0 import Priors

    sched = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    names = Priors.load(str(HERE / "priors" / "vd_questpro_novash_v0.json")).expressions
    ix = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(seed)
    base_rng = np.random.default_rng(0)  # the same "face" in every session; effort scales it
    dur = true_off + sched["duration"] + 40.0
    t = np.cumsum(rng.uniform(0.0095, 0.0125, int(dur / 0.0108)))
    X = np.abs(rng.normal(0, 0.004, (len(t), len(names))))
    truth = {}
    for n in sorted({n for p in sched["poses"] for n in p["targets"]}):
        truth[n] = min(float(base_rng.uniform(0.2, 0.85)) * effort, 1.0)
    # talking before and after the session: random jaw / smile / upper lip blips
    for a in np.arange(2.0, true_off - 3.0, 0.7):
        m = (t > a) & (t < a + 0.25)
        X[m, ix["JawOpen"]] += rng.uniform(0.05, 0.35)
        X[m, ix["MouthUpperUpLeft"]] += rng.uniform(0, 0.2)
    for p in sched["poses"]:
        if not p["targets"]:
            continue
        lag = rng.uniform(0.25, 0.6)
        a, b = true_off + p["go"] + lag, true_off + p["end"] + rng.uniform(-0.2, 0.3)
        m = (t > a) & (t < b)
        ramp = np.clip((t[m] - a) / 0.3, 0, 1)
        lead = p["targets"][0]
        for n in p["targets"]:
            if p["id"] in ("jaw_left", "mouth_left") and n.endswith("Right"):
                continue  # only one direction responds
            if p["id"] in ("jaw_right", "mouth_right") and n.endswith("Left"):
                continue
            X[m, ix[n]] = truth[n] * ramp + rng.normal(0, 0.006, m.sum())
        if p["id"] == "pucker":  # Quest Pro leak: pucker -> cheek puff
            X[m, ix["CheekPuffLeft"]] += 0.3 * truth[lead] * ramp
        if p["id"] == "lips_pressed_open":
            X[m, ix["JawOpen"]] = 0.5 * ramp
            X[m, ix["MouthClosed"]] = 0.4 * ramp
            truth["MouthClosed"] = 0.4
        # one-frame spike that must not count
        k = np.nonzero(m)[0]
        if len(k):
            X[k[len(k) // 2], ix[lead]] = 1.0
    X = np.clip(X, 0, 1)
    return Session(f"synthetic-{seed}", t, X, names, []), truth


def selftest() -> int:
    sched = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    true_off = 47.3
    s, truth = synthetic_session(7, true_off)
    off, best, second = align(s, sched)
    pr = measure(s, sched, off)
    err_off = abs(off - true_off)
    errs, bases = [], {}
    for n, v in truth.items():
        e = evidence(n, sched, pr, s.names, {})
        bases[e["basis"]] = bases.get(e["basis"], 0) + 1
        if e["basis"] == "measured" and n != "MouthClosed":
            errs.append(abs(e["target"] - v) / v)
    worst = max(errs)
    ratio = pr.ratio
    print(f"alignment: found {off:.2f} s, true {true_off:.2f} s (error {err_off * 1000:.0f} ms), match {best:.2f}, "
          f"next best {second:.2f}")
    print(f"basis counts {bases}; measured shapes worst relative error {worst * 100:.1f}% "
          f"(one-frame spikes of 1.0 were injected into every pose)")
    print(f"lips-pressed ratio: {ratio[0]:.3f} (true 0.800)" if ratio else "lips-pressed ratio: not measured")
    leak_ok = evidence("CheekPuffLeft", sched, pr, s.names, {})
    print(f"CheekPuffLeft: own pose {leak_ok['target']:.2f}, strongest elsewhere {leak_ok['other_pose']} "
          f"{leak_ok['other']:.2f} -> {leak_ok['basis']}")
    ok = bool(err_off < 0.15 and worst < 0.08 and ratio and abs(ratio[0] - 0.8) < 0.05)

    # end to end: two sessions (second one at 90% effort, other offset) through analyze() -> report + priors
    global load_session
    fakes = {"a": synthetic_session(7, 47.3), "b": synthetic_session(11, 12.0, effort=0.9)}
    real_loader = load_session
    load_session = lambda p: fakes[p][0]  # noqa: E731
    try:
        with tempfile.TemporaryDirectory() as td:
            pri, rep = Path(td) / "p.json", Path(td) / "r.md"
            analyze(["a", "b"], SCHEDULE_PATH, None, str(HERE / "priors" / "vd_questpro_novash_v0.json"),
                    str(pri), str(rep))
            d = json.loads(pri.read_text(encoding="utf-8"))
            text = rep.read_text(encoding="utf-8")
    finally:
        load_session = real_loader
    c0 = dict(zip(d["expressions"], d["c0"]))
    basis = d["meta"]["c0_basis"]
    t_a, t_b = fakes["a"][1], fakes["b"][1]
    geo = {n: math.sqrt(t_a[n] * t_b[n]) for n in t_a if basis.get(n) == "measured" and n != "MouthClosed"}
    e2e = max(abs(c0[n] - v) / v for n, v in geo.items())
    base = dict(zip(*[json.loads((HERE / "priors" / "vd_questpro_novash_v0.json").read_text())[k]
                      for k in ("expressions", "c0")]))
    leak_never_hotter = all(c0[n] >= min(base[n], 1.0) - 1e-9 for n, b in basis.items() if b.startswith("leak"))
    print(f"end to end: {len(geo)} measured shapes, worst error vs truth {e2e * 100:.1f}%, leak shapes never hotter: "
          f"{leak_never_hotter}, coupling ratio {d['coupled']['MouthClosed']['ratio']}, report "
          f"{len(text.splitlines())} lines, repeatability {'present' if '2 sessions' in text else 'MISSING'}")
    ok = (ok and e2e < 0.08 and leak_never_hotter and abs(d["coupled"]["MouthClosed"]["ratio"] - 1.25) < 0.07
          and "2 sessions" in text)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("guide")
    g.add_argument("--out-dir", default=r"E:\VRCFaceTracking-Adaptive\pose_guide")
    g.add_argument("--voice", default="Microsoft Zira Desktop")
    a = sub.add_parser("analyze")
    a.add_argument("recordings", nargs="+")
    a.add_argument("--schedule", default=None, help="guide timeline; default: try every pose_guide_v*.json")
    a.add_argument("--known-good")
    a.add_argument("--base-priors")
    a.add_argument("--out-priors")
    a.add_argument("--report")
    a.add_argument("--everyday", nargs="*", default=None,
                   help="normal-use recordings: ceilings never go below what the face reaches there")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    if args.cmd == "guide":
        wav, js = build_guide(Path(args.out_dir), args.voice)
        d = json.loads(js.read_text(encoding="utf-8"))["duration"]
        print(f"wrote {wav} ({d / 60:.1f} min) and {js}")
    elif args.cmd == "analyze":
        print(analyze(args.recordings, args.schedule, args.known_good, args.base_priors, args.out_priors,
                      args.report, args.everyday))
    else:
        sys.exit(selftest())


if __name__ == "__main__":
    main()
