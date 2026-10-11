"""Golden test for the C# guided-calibration ring (VRCFaceTracking.Core/Adaptive/Guided/PoseRing.cs).

Runs ring_sim's stress loop (revised rules) over recorded pose sessions, hands every step's inputs to the C# ring
(AdaptiveReplay ring ...) and requires the same result per half: when it filled, the fill reached, the open-loop
"would ask" verdict and reason, the weak flag, the recorded level and the recorded / analysis ratio.

    python golden_ring.py REC.vftrec [REC2 ...]
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
from ring_sim import HZ, stress  # noqa: E402

TOL = 1e-9


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--first-ref", default=str(HERE / "priors" / "vd_questpro_novash_v0.json"))
    ap.add_argument("--repeat-ref", default=str(HERE / "priors" / "vd_questpro_novash_v1.json"))
    a = ap.parse_args(argv)

    cases = []
    names_box = {}

    def collect(c):
        cases.append(c)

    from ring_sim import find_runs  # names come from the recording
    names_box["names"] = find_runs(a.recordings[:1])[0].s.names
    stress(a.recordings, a.first_ref, a.repeat_ref, None, on_case=collect)
    names = names_box["names"]

    build()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        manifest = {"names": names, "dt": 1.0 / HZ, "cases": []}
        with open(d / "data.bin", "wb") as f:
            for i, c in enumerate(cases):
                G = np.ascontiguousarray(c["G"], dtype="<f8")
                manifest["cases"].append({
                    "id": i, "pose": c["step"].id, "halves": c["step"].halves, "companions": sorted(c["companions"]),
                    "ref": c["ref"].tolist(), "tref": c["tref"].tolist(), "floor": c["floor"].tolist(),
                    "jaw_raw": c["jaw_raw"], "n": len(G), "offset": f.tell()})
                f.write(G.tobytes())
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["dotnet", str(DLL), "ring", str(d / "manifest.json"), str(d / "data.bin"), str(d / "out.json")],
                       check=True, stdout=subprocess.DEVNULL)
        out = json.loads((d / "out.json").read_text(encoding="utf-8"))["cases"]

    bad = []
    halves_checked = 0
    for c, o in zip(cases, out):
        exp = c["result"]
        got = {h["name"]: h for h in o["halves"]}
        for name, e in exp.items():
            g = got.get(name)
            halves_checked += 1
            where = f"{c['scenario']} / {c['reference']} / session {c['session'] + 1} / {c['step'].id} {name}".rstrip()
            if g is None:
                bad.append(f"{where}: half missing")
                continue
            for key in ("filled_s", "need_s", "level", "rec_ratio"):
                if not close(e[key], g[key]):
                    bad.append(f"{where}: {key} Python {e[key]} vs C# {g[key]}")
            for key in ("reason", "weak"):
                if e[key] != g[key]:
                    bad.append(f"{where}: {key} Python {e[key]} vs C# {g[key]}")

    print(f"{len(cases)} pose cases, {halves_checked} halves compared, {len(bad)} differences")
    for b in bad[:25]:
        print("  " + b)
    print("GOLDEN PASS" if not bad else "GOLDEN FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
