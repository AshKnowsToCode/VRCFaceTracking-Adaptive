"""Golden test for the C# guided-calibration analysis (VRCFaceTracking.Core/Adaptive/Guided/PoseMeasure.cs,
PoseAnalysis.cs) against tools/replay/poses.py.

    python golden_analysis.py POSE_SESSIONS.vftrec --everyday REC.vftrec ... [--base-priors priors/vd_questpro_novash_v0.json]
        [--expect priors/vd_questpro_novash_v1.json]

1. Measurement: every pose window of every guide run found in the recording goes through PoseMeasure; the held levels,
   rest percentiles and lips ratio must equal poses.measure().
2. Analysis: poses.measure()'s results go through PoseAnalysis with the same "now" and everyday levels as
   poses.analyze(); the ceilings, their basis and the coupling ratio must equal what analyze() writes.
3. --expect: analyze() today must still reproduce that priors file (the values in use).
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
from poses import (LATE_S, REACT_S, _grid, analyze, everyday_held, find_sessions, header_calibration,  # noqa: E402
                   load_session, measure)

HZ = 100.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording")
    ap.add_argument("--everyday", nargs="+", default=[])
    ap.add_argument("--base-priors", default=str(HERE / "priors" / "vd_questpro_novash_v0.json"))
    ap.add_argument("--expect", default=str(HERE / "priors" / "vd_questpro_novash_v1.json"))
    a = ap.parse_args(argv)

    s = load_session(a.recording)
    names = s.names
    scheds = [json.loads(c.read_text(encoding="utf-8")) for c in sorted(HERE.glob("pose_guide_v*.json"))]
    found = find_sessions(s, scheds)
    runs = [(use, off, measure(s, use, off)) for use, off, *_ in found]
    print(f"{Path(a.recording).name}: {len(runs)} guide run(s) at " + ", ".join(f"{off:.0f} s" for _, off, _ in runs))
    build()
    problems = []

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # 1. measurement
        g, G = _grid(s, HZ)
        manifest = {"names": names, "dt": 1.0 / HZ, "sessions": []}
        with open(d / "frames.bin", "wb") as f:
            for use, off, _ in runs:
                poses = []
                for p in use["poses"]:
                    lo, hi = off + p["go"] + REACT_S, off + p["end"] + LATE_S
                    rows = np.ascontiguousarray(G[(g >= lo) & (g < hi)], dtype="<f8")
                    poses.append({"id": p["id"], "relax": not p["targets"], "n": len(rows), "offset": f.tell()})
                    f.write(rows.tobytes())
                manifest["sessions"].append({"poses": poses})
        (d / "frames.json").write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["dotnet", str(DLL), "measure", str(d / "frames.json"), str(d / "frames.bin"),
                        str(d / "measured.json")], check=True, stdout=subprocess.DEVNULL)
        measured = json.loads((d / "measured.json").read_text(encoding="utf-8"))["sessions"]
        worst = 0.0
        for k, ((_, off, pr), cs) in enumerate(zip(runs, measured)):
            if set(cs["held"]) != set(pr.held):
                problems.append(f"run {k + 1}: poses measured differ: {sorted(set(cs['held']) ^ set(pr.held))}")
            for pid, v in pr.held.items():
                if pid in cs["held"]:
                    worst = max(worst, float(np.max(np.abs(np.array(cs["held"][pid]) - v))))
            p95 = np.array([pr.rest_p95[n] for n in names])
            worst = max(worst, float(np.max(np.abs(np.array(cs["rest_p95"]) - p95))))
            if (pr.ratio is None) != (cs["ratio"] is None):
                problems.append(f"run {k + 1}: lips ratio Python {pr.ratio} vs C# {cs['ratio']}")
            elif pr.ratio is not None:
                worst = max(worst, abs(cs["ratio"][0] - pr.ratio[0]), abs(cs["ratio"][1] - pr.ratio[1]))
        print(f"1. measurement: {sum(len(pr.held) for *_, pr in runs)} poses, largest difference {worst:.1e}")
        if worst > 1e-12:
            problems.append(f"measurement differs by {worst:.1e}")

        # 2. analysis, with analyze()'s own inputs
        now = header_calibration(s.header, names) or {}
        everyday = everyday_held(a.everyday, names) if a.everyday else {}
        out_priors = d / "analyzed.json"
        analyze([a.recording], None, None, a.base_priors, str(out_priors), None, a.everyday or None)
        expect_py = json.loads(out_priors.read_text(encoding="utf-8"))
        inp = {
            "names": names,
            "poses": [{"id": p["id"], "targets": p["targets"]} for p in scheds[-1]["poses"]],
            "runs": [{"held": {pid: v.tolist() for pid, v in pr.held.items()},
                      "rest_p95": [pr.rest_p95[n] for n in names], "ratio": list(pr.ratio) if pr.ratio else None}
                     for *_, pr in runs],
            "now": [now.get(n, 0.0) for n in names],
            "everyday": [everyday.get(n, 0.0) for n in names],
        }
        (d / "analyze_in.json").write_text(json.dumps(inp), encoding="utf-8")
        subprocess.run(["dotnet", str(DLL), "analyze", str(d / "analyze_in.json"), str(d / "analyze_out.json")],
                       check=True, stdout=subprocess.DEVNULL)
        cs = json.loads((d / "analyze_out.json").read_text(encoding="utf-8"))
        ix = {n: i for i, n in enumerate(expect_py["expressions"])}
        mism = []
        for n, v in cs["shapes"].items():
            py_c = expect_py["c0"][ix[n]]
            if abs(round(v["c"], 6) - py_c) > 1e-12:
                mism.append(f"{n} c Python {py_c} vs C# {v['c']}")
            if v["basis"] != expect_py["meta"]["c0_basis"][n]:
                mism.append(f"{n} basis Python {expect_py['meta']['c0_basis'][n]} vs C# {v['basis']}")
        py_cr = expect_py["coupled"].get("MouthClosed", {}).get("ratio")
        cs_cr = round(cs["coupling"], 4) if cs["coupling"] is not None else None
        if py_cr != cs_cr:
            mism.append(f"coupling Python {py_cr} vs C# {cs_cr}")
        print(f"2. analysis: {len(cs['shapes'])} ceilings + coupling ratio {cs_cr}, {len(mism)} differences")
        problems += mism

        # 3. still reproduces the values in use
        if a.expect:
            exp = json.loads(Path(a.expect).read_text(encoding="utf-8"))
            same = (exp["expressions"] == expect_py["expressions"] and exp["c0"] == expect_py["c0"]
                    and exp["coupled"] == expect_py["coupled"])
            print(f"3. today's analyze() reproduces {Path(a.expect).name}: {'yes' if same else 'NO'}")
            if not same:
                problems.append(f"analyze() no longer reproduces {Path(a.expect).name}")

    for p in problems[:25]:
        print("  " + p)
    print("GOLDEN PASS" if not problems else "GOLDEN FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
