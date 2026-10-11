"""Golden test for the C# port of the v0 calibrator (DESIGN.md section 10).

Replays recordings through adaptive_v0.py (the reference) and through the C# AdaptiveCalibrator
(VRCFaceTracking.Core/Adaptive/Calibration, run by AdaptiveReplay/) and requires the outputs to match.

    python golden_v0.py REC.vftrec [REC2 ...] [--priors priors/vd_questpro_novash_v1.json] [--split 0.5] [--tol 1e-4]

Per recording: every calibrated value (max |dy| <= tol), the gate per frame, and the learned ceilings / guard
factors at the end. --split also saves the learned state halfway (JSON) and continues on a fresh calibrator, in
both languages, which checks the save format round trip.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from adaptive_v0 import AdaptiveCalibrator, Priors  # noqa: E402
from vftrec import read  # noqa: E402

PROJECT = HERE / "AdaptiveReplay" / "AdaptiveReplay.csproj"
DLL = HERE / "AdaptiveReplay" / "bin" / "Release" / "net10.0" / "AdaptiveReplay.dll"


def build():
    subprocess.run(["dotnet", "build", str(PROJECT), "-c", "Release", "-nologo", "-v", "q"], check=True,
                   stdout=subprocess.DEVNULL)


def run_cs(rec: str, priors: str, split: int | None, module: str | None = None):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.bin"
        source = ["--module", module] if module else ["--priors", priors]
        cmd = ["dotnet", str(DLL), *source, "--out", str(out), rec]
        if split is not None:
            cmd[2:2] = ["--split", str(split)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        b = out.read_bytes()
    n, k = struct.unpack_from("<ii", b, 0)
    pos = 8
    y = np.frombuffer(b, "<f8", n * k, pos).reshape(n, k)
    pos += 8 * n * k
    gated = np.frombuffer(b, "u1", n, pos).astype(bool)
    pos += n
    c = np.frombuffer(b, "<f8", k, pos)
    pos += 8 * k
    g = np.frombuffer(b, "<f8", k, pos)
    pos += 8 * k
    (active,) = struct.unpack_from("<d", b, pos)
    return y, gated, c, g, active


def run_py(r, priors: Priors, split: int | None):
    nch = len(r.header["frame"]["channels"])
    X = r.raw[:, nch:].astype(float)
    t = r.t
    cal = AdaptiveCalibrator(priors)
    y = np.empty_like(X)
    gated = np.zeros(len(X), bool)
    for f in range(len(X)):
        if f == split:
            s = json.loads(json.dumps(cal.state()))
            cal = AdaptiveCalibrator(priors)
            cal.load_state(s)
        y[f] = cal.step(t[f], X[f])
        gated[f] = cal.gated
    return y, gated, cal.c.copy(), cal.g.copy(), cal.active_s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--priors", default=str(HERE / "priors" / "vd_questpro_novash_v1.json"))
    ap.add_argument("--split", type=float, default=None, help="save + reload the state at this fraction of frames")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--module", help="C# side uses the priors embedded in the app for this module (e.g. 'Virtual Desktop')")
    a = ap.parse_args(argv)

    build()
    ok_all = True
    for path in a.recordings:
        r = read(path)
        names = r.expressions
        pri = Priors.load(a.priors).reorder(names)
        n = len(r.t)
        split = int(n * a.split) if a.split is not None else None
        yp, gp, cp, gpp, ap_ = run_py(r, pri, split)
        yc, gc, cc, gcc, ac = run_cs(path, a.priors, split, a.module)
        if yc.shape != yp.shape:
            print(f"{Path(path).name}: FAIL frame count C# {yc.shape} vs Python {yp.shape}")
            ok_all = False
            continue
        dy = np.abs(yc - yp)
        f, i = np.unravel_index(int(np.argmax(dy)), dy.shape)
        gate_mis = int((gc != gp).sum())
        dc = float(np.max(np.abs(cc / cp - 1)))
        dg = float(np.max(np.abs(gcc / gpp - 1)))
        changed = int((np.abs(cp / pri.c0.clip(0.15, 1.0) - 1) > 0.01).sum())
        ok = dy.max() <= a.tol and gate_mis == 0 and dc <= a.tol and dg <= a.tol and abs(ac - ap_) <= 1e-6
        ok_all &= ok
        print(f"{Path(path).name}{' (split at frame ' + str(split) + ')' if split is not None else ''}: {n} frames, "
              f"{ap_:.0f} active s, {int(gp.sum())} gated frames, {changed} ceilings moved | max |dy| {dy.max():.2e} "
              f"(frame {f}, {names[i]}; {int((dy > 1e-9).sum())} values differ by > 1e-9), gate mismatches "
              f"{gate_mis}, ceilings {dc:.1e}, guard {dg:.1e}, "
              f"active {abs(ac - ap_):.1e} -> {'PASS' if ok else 'FAIL'}")
    print("GOLDEN PASS" if ok_all else "GOLDEN FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
