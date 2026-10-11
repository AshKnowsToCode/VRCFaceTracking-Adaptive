"""Builds the guided calibration's per-module pose data from recorded pose sessions (DESIGN.md 8.1):
per pose the expressions that move with it (companions, both sides; never "wrong expression") and a typical
"best in this pose" value per asked-for expression (the ring's reference on a person's first run).

    python guided_module_data.py POSE_SESSIONS.vftrec --module "Virtual Desktop" --priors priors/vd_questpro_novash_v1.json
        --out ../../VRCFaceTracking.Core/Adaptive/Guided/Data/guided-virtual-desktop.json

Companions = union over the sessions of ring_sim.companions (>= 0.3 of the channel ceiling during the pose, mirrored,
plus the pose's own STEP_COMPANIONS). Typical = geometric mean over the sessions of the best 0.5 s hold, using only
holds clearly above the relaxed face and >= 0.25 of the ceiling (ring_sim.pose_ref's rule); none -> the app uses the
channel ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from adaptive_v0 import Priors  # noqa: E402
from ring_sim import HELD_S, HZ, SCENARIOS, build_steps, companions, find_runs, prepare, trailing  # noqa: E402

# The common wrong face of a pose must stay "wrong expression", even if the sessions the data comes from made it
# (nov-ash 2026-10-10: the grimace bared the teeth, the press rolled the lips in, the "oh" was a kiss, the jaw-forward
# pushed the pout): these never become companions.
MISTAKES = {
    "stretch": ("MouthLowerDown", "MouthUpperUp", "MouthUpperDeepen"),
    "press": ("LipSuck",),
    "funnel": ("LipPucker",),
    "jaw_forward": ("MouthRaiser",),
    "cheek_puff": ("LipPucker",),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--module", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    runs = find_runs(a.recordings)
    names = runs[0].s.names
    ref = Priors.load(a.priors).reorder(names).c0
    ix = {n: i for i, n in enumerate(names)}
    prepped = [prepare(r, SCENARIOS[0][1]) for r in runs]
    poses = {}
    for st in build_steps():
        comp = set()
        best = {n: [] for n in st.targets}
        for segs, floor in prepped:
            comp |= companions(st, segs, names, ref)
            G = np.concatenate([segs[p] for p in st.sources])
            held = trailing(G, int(HELD_S * HZ), np.min).max(axis=0)
            for n in st.targets:
                c = ix[n]
                if held[c] >= max(floor[c], 0.25 * ref[c]):
                    best[n].append(float(held[c]))
        comp = {c for c in comp if not c.startswith(MISTAKES.get(st.id, ()))}
        typical = {n: round(float(np.exp(np.mean(np.log(v)))), 6) for n, v in best.items() if v}
        poses[st.id] = {"companions": sorted(comp), "typical": typical}

    out = {
        "schema": 1,
        "module": a.module,
        "source": "pose sessions " + "; ".join(r.label for r in runs) + f"; ceilings {Path(a.priors).name}",
        "note": "Companions: expressions that move with the pose (>= 0.3 of their ceiling, both sides), never "
                "'wrong expression'. Typical: geometric mean of the best 0.5 s hold in the pose (the ring's reference "
                "on a first run). One person so far; replace with volunteer data (DESIGN 8.4).",
        "poses": poses,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(poses)} poses -> {a.out}")
    for pid, d in poses.items():
        print(f"  {pid:18s} companions {len(d['companions']):2d}, typical " +
              ", ".join(f"{n} {v:.2f}" for n, v in d["typical"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
