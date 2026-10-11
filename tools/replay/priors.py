"""Build a v0 priors file for adaptive_v0.py.

Ceiling prior c0: one person's long-term stock-calibration maxima (a LocalSettings.json whose calibration is known to
look right), floored so a noise-level max can't become a 7-20x gain. Shapes with no evidence (max 0) get c0 = 1,
i.e. no scaling until the calibrator sees evidence. Rest b0 / noise sigma0: robust statistics of the raw module
output in recordings (frames gated as headset-off are skipped).

    python priors.py --known-good LocalSettings.json --out priors/vd_questpro_novash_v0.json REC.vftrec ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adaptive_v0 import Priors  # noqa: E402
from stock import load_calibration  # noqa: E402
from vftrec import read  # noqa: E402

C0_FLOOR = 0.15  # known-good maxima below this were noise-level (mouth sideways 0.05, cheek puff 0.09)
SIGMA_FLOOR = 0.002
COUPLED = {"MouthClosed": "JawOpen"}  # child is defined relative to parent (MouthClosed == JawOpen = lips sealed)


def worn_mask(rec, frozen_s=0.5, eps=1e-4) -> np.ndarray:
    """False on all-zero frames and on runs where no expression moved for frozen_s (headset off)."""
    nch = len(rec.header["frame"]["channels"])
    x = rec.raw[:, nch:]
    t = rec.t
    moved = np.r_[True, np.abs(np.diff(x, axis=0)).max(axis=1) > eps]
    last = np.maximum.accumulate(np.where(moved, np.arange(len(t)), 0))
    return (x.max(axis=1) > 0) & ((t - t[last]) < frozen_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--known-good", required=True, help="LocalSettings.json with a trusted stock calibration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--module", default="Virtual Desktop")
    ap.add_argument("--headset", default="Quest Pro")
    a = ap.parse_args(argv)

    recs = [read(p) for p in a.recordings]
    names = recs[0].expressions
    nch = len(recs[0].header["frame"]["channels"])
    x = np.concatenate([r.raw[worn_mask(r), nch:] for r in recs]).astype(float)

    mx, _, _, _ = load_calibration(a.known_good, names)
    c0 = np.where(mx > 0, np.maximum(mx, C0_FLOOR), 1.0)
    q05, q50 = np.percentile(x, 5, axis=0), np.percentile(x, 50, axis=0)
    sigma0 = np.empty(len(names))
    for i in range(len(names)):
        low = x[x[:, i] <= q50[i], i]
        sigma0[i] = 1.4826 * np.median(np.abs(low - np.median(low))) if len(low) else 0.0
    sigma0 = np.maximum(sigma0, SIGMA_FLOOR)

    ix = {n: i for i, n in enumerate(names)}
    coupled = {}
    for child, parent in COUPLED.items():
        # keep the relative gain the known-good calibration gave the child over its parent (fixed, never learned)
        ratio = c0[ix[parent]] / c0[ix[child]] if mx[ix[child]] > 0 and mx[ix[parent]] > 0 else 1.0
        coupled[child] = {"parent": parent, "ratio": round(float(ratio), 4)}

    meta = {"name": Path(a.out).stem, "module": a.module, "headset": a.headset,
            "c0_source": str(a.known_good), "c0_floor": C0_FLOOR,
            "rest_source": [Path(p).name for p in a.recordings], "worn_frames": int(len(x)),
            "note": "v0: one person's data (nov-ash). Replace with volunteer priors (DESIGN 8.4)."}
    Priors(names, c0, q05, sigma0, coupled, meta).save(a.out)
    floored = [n for n in names if 0 < mx[ix[n]] < C0_FLOOR]
    print(f"wrote {a.out}: {len(names)} shapes, {int((mx > 0).sum())} with a known-good max, "
          f"floored to {C0_FLOOR}: {floored}; coupled {coupled}")


if __name__ == "__main__":
    main()
