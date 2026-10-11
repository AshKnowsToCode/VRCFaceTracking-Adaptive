"""End-to-end check of the C# guided calibration (GuidedSession) on a recorded pose session.

    python guided_replay.py POSE_SESSIONS.vftrec [--run 2] [--everyday REC.vftrec ...]

Plays each library pose's recorded window to the session as if the person did it when asked (over and over until the
ring fills; the simulated person skips whenever the session asks) and prints what happened, the summary, and how the
resulting starting values compare with priors/vd_questpro_novash_v1.json (built from both sessions with poses.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from golden_v0 import DLL, build  # noqa: E402
from poses import everyday_held  # noqa: E402
from ring_sim import SCENARIOS, build_steps, find_runs, prepare  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("--run", type=int, default=2, help="which guide run in the recording (1-based)")
    ap.add_argument("--priors", default=str(HERE / "priors" / "vd_questpro_novash_v1.json"))
    ap.add_argument("--everyday", nargs="*", default=[])
    ap.add_argument("--redo", nargs="*", default=[], help="pose ids to redo from the summary, in order")
    ap.add_argument("--save-priors", help="write the run's new starting values (what Keep would apply) here")
    a = ap.parse_args(argv)

    runs = find_runs([a.recording])
    run = runs[a.run - 1]
    names = run.s.names
    segs, _ = prepare(run, SCENARIOS[0][1])
    windows = {"relax": segs["relax"], "relax_end": segs["relax_end"]}
    for st in build_steps():
        windows[st.id] = np.concatenate([segs[p] for p in st.sources])
    everyday = everyday_held(a.everyday, names) if a.everyday else {}

    build()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        manifest = {"names": names, "priors": a.priors, "everyday": [everyday.get(n, 0.0) for n in names],
                    "windows": {}, "redo": a.redo}
        with open(d / "frames.bin", "wb") as f:
            for pid, G in windows.items():
                manifest["windows"][pid] = {"n": len(G), "offset": f.tell()}
                f.write(np.ascontiguousarray(G, dtype="<f8").tobytes())
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["dotnet", str(DLL), "guided", str(d / "manifest.json"), str(d / "frames.bin"),
                        str(d / "out.json")], check=True, stdout=subprocess.DEVNULL)
        out = json.loads((d / "out.json").read_text(encoding="utf-8"))

    if a.save_priors:
        Path(a.save_priors).write_text(out["priors_json"], encoding="utf-8")
    print(f"{run.label}: the whole guided run took {out['seconds']:.0f} s of tracking frames")
    for p in out["poses"]:
        state = "skipped" if p["skipped"] else f"filled in {p['filled_s']:.1f} s" if p["filled_s"] else "done"
        asks = f", asked: {', '.join(p['asks'])}" if p["asks"] else ""
        print(f"  {p['id']:18s} {state}{' (weak)' if p['weak'] else ''}{asks}")
    print("Summary:")
    for s in out["summary"]:
        print("  " + s)
    if len(out["summaries"]) > 1:
        first, last = out["summaries"][0], out["summaries"][-1]
        print(f"Redo {', '.join(a.redo)}: rows that changed")
        for before, after in zip(first, last):
            if before != after:
                print(f"  {before}  ->  {after}")
    v1 = json.loads(Path(a.priors).read_text(encoding="utf-8"))
    ix = {n: i for i, n in enumerate(v1["expressions"])}
    ratios = {n: v["c"] / v1["c0"][ix[n]] for n, v in out["shapes"].items()}
    r = np.array(list(ratios.values()))
    print(f"Starting values vs {Path(a.priors).name} (both sessions): {len(r)} expressions, median ratio "
          f"{np.median(r):.2f}, within 10%: {int((np.abs(np.log(r)) < np.log(1.1)).sum())}, "
          f"coupling {out['coupling']}")
    far = sorted(ratios.items(), key=lambda kv: -abs(np.log(kv[1])))[:8]
    print("  largest differences: " + ", ".join(f"{n} {v:.2f}x" for n, v in far))
    return 0


if __name__ == "__main__":
    sys.exit(main())
