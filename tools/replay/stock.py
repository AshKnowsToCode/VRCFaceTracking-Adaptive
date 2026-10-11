"""Exact emulators of the stock VRCFT mutations that touch expressions, vectorised over the 88 shapes.

Used as reference candidates by evaluate.py and to check the replay harness itself: feeding a recording's raw block
through StockCalibration + correctors() reproduces its out block (verified on 20261008-214509.vftrec, all 88 shapes,
with the three resets found in it).

Source: VRCFaceTracking.Core/Params/Data/Mutation/Calibration.cs and Correctors.cs (6.0, master 6432e6a).
"""

from __future__ import annotations

import numpy as np

POINTS = 64
STEP = np.float32(0.15) * np.float32(0.1)  # sDelta * dT, dT hard-coded to 0.1 in MutateData


class StockCalibration:
    """Calibration.cs, one instance per run. State per shape: max, progress, sample window, step tracker.

    `max_` and `progress` are what LocalSettings.json saves. The sample counter is NOT saved, so `warm=False`
    (the default) reproduces an app start: progress drops to 1/64 on the first sample and re-ramps. `warm=True`
    reproduces a recording that started while the app was already running (counter = progress * 64).
    """

    def __init__(self, n: int, max_=None, progress=None, blend: float = 1.0, continuous: bool = False,
                 warm: bool = False):
        self.n = n
        self.blend = blend
        self.continuous = continuous
        self.max = np.zeros(n) if max_ is None else np.asarray(max_, dtype=float).copy()
        self.progress = np.zeros(n) if progress is None else np.asarray(progress, dtype=float).copy()
        # warm = the app was already running (recording started mid-session): the counter matches progress.
        self.fixed = np.rint(self.progress * POINTS).astype(int) if warm else np.zeros(n, dtype=int)
        self.finished = np.zeros(n, dtype=bool)
        self.roll = np.zeros(n, dtype=int)
        self.step = np.zeros(n)
        self.data = np.zeros((n, POINTS))

    def reset(self):
        """The "Reset Calibration" button."""
        self.__init__(self.n, blend=self.blend, continuous=self.continuous)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        sample = np.abs(v - self.step) >= STEP
        grow = sample & (self.fixed < POINTS)
        self.fixed[grow] += 1
        self.progress[grow] = self.fixed[grow] / POINTS
        self.finished |= sample & ~grow
        idx = np.nonzero(sample)[0]
        self.data[idx, self.roll[idx]] = v[idx]
        upd = sample & (~self.finished | self.continuous)
        u = np.nonzero(upd)[0]
        self.roll[u] = (self.roll[u] + 1) % POINTS
        stats = u[self.fixed[u] >= 0.1 * POINTS]
        if len(stats):
            self.max[stats] = np.maximum(self.max[stats], self.data[stats].max(axis=1))
        self.step = np.floor(v / STEP) * STEP

        conf = self.blend * self.progress
        with np.errstate(divide="ignore", invalid="ignore"):
            out = conf * (v / self.max) + (1 - conf) * v
        return np.where((self.max == 0) | ~np.isfinite(out), v, out)


def correctors(y: np.ndarray, names: list[str], mouth_closed_fix=True, lip_suck_fix=True) -> np.ndarray:
    """Correctors.cs expression part (eyelid blend / gaze symmetrise act on eye channels, not here)."""
    y = y.copy()
    ix = {n: i for i, n in enumerate(names)}
    if mouth_closed_fix:
        y[..., ix["MouthClosed"]] = np.minimum(y[..., ix["MouthClosed"]], y[..., ix["JawOpen"]])
    if lip_suck_fix:
        for side in ("Left", "Right"):
            y[..., ix[f"LipSuckLower{side}"]] *= 1 - y[..., ix[f"MouthLowerDown{side}"]]
            y[..., ix[f"LipSuckUpper{side}"]] *= 1 - y[..., ix[f"MouthUpperUp{side}"]]
    return y


def load_calibration(local_settings_path: str, names: list[str]):
    """(max, progress, blend, continuous) from a VRCFT LocalSettings.json, in `names` order."""
    import json

    d = json.load(open(local_settings_path, encoding="utf-8-sig"))
    c = json.loads(d["Calibration"])
    shapes = {s["name"]: s for s in c["calData"]["Shapes"]}
    mx = np.array([shapes[n]["max"] if n in shapes else 0.0 for n in names])
    pr = np.array([shapes[n]["progress"] if n in shapes else 0.0 for n in names])
    return mx, pr, float(c.get("calibrationBlend", 1.0)), bool(c.get("continuousCalibration", False))
