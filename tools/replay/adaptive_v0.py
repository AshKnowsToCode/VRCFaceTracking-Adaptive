"""Adaptive calibration v0 - Python reference implementation (DESIGN.md section 6, v0 scope in section 10).

One `AdaptiveCalibrator` replaces the stock Calibration mutation. Per expression it learns a rest level and noise
(-> soft deadzone) and a ceiling from strong *held* peaks only, shrunk toward a prior and bounded around it
(v0: the ceiling may only rise, at most 1.6x, so passive learning can make a shape calmer but never hotter),
rate-limited and gated. Shapes defined relative to another shape (MouthClosed is relative to JawOpen) are never
scaled on their own: they take their parent's scale times a fixed ratio from the priors.

Written for a straight port to C# (Phase 3): explicit state, one `step(t, x)` per module frame, no look-ahead.
Everything is vectorised over the expressions; the only per-shape Python loops run on rare events (an episode
ending, the 0.5 s ceiling update).

    cal = AdaptiveCalibrator(Priors.load("priors/vd_questpro_novash_v0.json"))
    y = cal.step(t_seconds, x)      # x: raw expression weights in module order, y: calibrated
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

LN2 = math.log(2.0)


@dataclass
class Params:
    # 6.1 gating (output is still produced; learning is skipped)
    frozen_eps: float = 1e-4  # no expression moved more than this ...
    frozen_s: float = 0.5  # ... for this long -> frozen (headset off / VD paused)
    jump_count: int = 12  # this many expressions ...
    jump_size: float = 0.3  # ... jumping this much in one frame -> implausible
    jump_pause_s: float = 1.0
    recover_s: float = 2.0  # learning stays off this long after zeros / frozen end
    max_frame_dt: float = 0.1  # the active clock never advances more than this per frame

    # 6.2 rest and noise
    hist_bins: int = 200
    hist_hz: float = 20.0
    hist_half_life_s: float = 600.0  # 10 active minutes
    rest_q: float = 0.05
    rest_min_samples: int = 1200  # 60 active s at 20 Hz before the data replaces the priors
    rest_max_spread: float = 0.25  # Q50 - Q05 above this -> not a rest-dominated shape, keep priors
    deadzone_k: float = 3.0
    sigma_floor: float = 0.002
    # The lower-half MAD measures the face, not the sensor, when a shape is rarely at rest: in nov-ash's social sessions
    # (2026-10-13..17, whole mouth at rest in ~1% of frames) smile, upper lip, dimple and jaw learned 0.04-0.10, and the
    # deadzones (up to 0.29 raw) ate gentle expressions. Quest Pro noise at rest is <= 0.016. So the noise is bounded
    # like every other learned value (deadzone <= 0.03).
    sigma_max: float = 0.01
    # The rest level rises at most this far above its prior: a face held for most of 10+ minutes would otherwise
    # become the new neutral and fade.
    rest_max_rise: float = 0.05
    update_every_s: float = 0.5  # quantiles, ceilings and guardrail are recomputed on this active cadence

    # 6.3 ceiling
    hold_s: float = 0.15  # held signal = rolling min over this window
    theta_k: float = 4.0  # episode threshold = b + theta_k * sigma + theta_add
    theta_add: float = 0.05
    episode_min_s: float = 0.3
    peaks_keep: int = 16
    peak_half_life_s: float = 20 * 3600.0  # 20 active hours
    k_weight: float = 2.5  # c_obs = the peak where the accumulated weight reaches this (~3rd highest)
    # Headroom: c_target = eta * c_obs. The design's 0.9 makes the ~3rd-highest held peak reach 111%; on nov-ash's
    # replay that walked jaw / cheeks / brows down to the bound (25% hotter) within 40 active minutes. v0 uses 1.0.
    eta: float = 1.0
    n0: float = 6.0  # prior strength, in peaks
    n_eff_max: float = 30.0
    # c stays in [c0 / bound_down, c0 * bound_up]. Lowering c makes a shape HOTTER - the direction every stock
    # calibration incident went - and passive data cannot tell a subtle face from a weak sensor (only max-effort
    # poses can). So v0's passive learning may only RAISE c (protects against pinning when you go past the prior);
    # lowering waits for guided-boost evidence (DESIGN 8.1). bound_down > 1 re-enables it.
    bound_down: float = 1.0
    bound_up: float = 1.6
    # Raising c (calmer) is the safe direction and fixes pinning, so it follows the ~3rd-highest held peak quickly
    # and without shrinkage. Lowering (hotter) is shrunk toward the prior and slow.
    slew_up_per_min: float = 0.50  # c rises at most this fraction per active minute
    slew_per_10min: float = 0.10  # c falls at most this fraction per 10 active minutes
    c_abs_min: float = 0.15  # never more than ~6.7x gain, whatever the prior says
    min_denominator: float = 0.05

    # 6.4 saturation guardrail. On in v0: it only ever makes a shape calmer (g >= 1, c*g <= c0 * bound_up), and on
    # the 10-loop replay it brought speech pinning from 0.9% to 0.5% (= the stock known-good values).
    guardrail: bool = True
    sat_level: float = 0.98
    sat_run_max_s: float = 0.5  # only short pinned runs count (speech / flicker, not deliberate holds)
    guard_window_s: float = 600.0
    guard_share: float = 0.03
    guard_step: float = 1.05
    guard_every_s: float = 30.0
    guard_half_life_s: float = 3600.0  # g relaxes back to 1 over about an active hour (10 h kept old calming for days)
    # A step needs short pins since the previous step: the share is a 10-minute memory, so one burst kept stepping
    # every 30 s on the same evidence (mouth press 1.0 -> 1.37 on 2026-10-17, no new pin after the first step) ...
    guard_needs_new_pins: bool = True
    # ... and short pins in at least guard_min_minutes different active minutes of the last guard_minutes (0 = off).
    guard_min_minutes: int = 3
    guard_minutes: int = 10
    # The share of short pins is measured against the time the shape is shown at >= this level (0 = time above theta,
    # the old rule). Theta includes 4 sigma, so once sigma was capped it fell from ~0.4 to ~0.1 for busy shapes and the
    # share got diluted: dimple / upper lip pinned in laughs (2026-10-13 replay: added pinning 0.89% -> 1.12% of
    # talking frames with theta, 0.62% with 0.5).
    guard_busy_level: float = 0.5


@dataclass
class Priors:
    """Per-module starting values. Unknown module -> `Priors.identity(names)` (behaves like calibration off)."""

    expressions: list[str]
    c0: np.ndarray  # ceiling prior (raw value that maps to 100%)
    b0: np.ndarray  # rest prior
    sigma0: np.ndarray  # noise prior
    coupled: dict = field(default_factory=dict)  # child -> {"parent": name, "ratio": r}
    meta: dict = field(default_factory=dict)
    # Optional per-expression upper limit for the learned ceiling, above c0 * bound_up (0 = none). A guided calibration
    # that makes an expression stronger stores the ceiling that was in effect before, so normal use can calm it all the
    # way back if everyday expressions go past the new value (DESIGN 8.1).
    c_max: np.ndarray | None = None

    @classmethod
    def identity(cls, names: list[str]) -> "Priors":
        n = len(names)
        return cls(list(names), np.ones(n), np.zeros(n), np.full(n, 0.005),
                   {"MouthClosed": {"parent": "JawOpen", "ratio": 1.0}}, {"name": "identity"})

    @classmethod
    def load(cls, path: str) -> "Priors":
        d = json.load(open(path, encoding="utf-8"))
        names = d["expressions"]
        return cls(names, np.array(d["c0"], float), np.array(d["b0"], float), np.array(d["sigma0"], float),
                   d.get("coupled", {}), d.get("meta", {}),
                   np.array(d["c_max"], float) if d.get("c_max") is not None else None)

    def save(self, path: str):
        d = {"schema": 1, "meta": self.meta, "expressions": self.expressions,
             "c0": [round(float(v), 6) for v in self.c0], "b0": [round(float(v), 6) for v in self.b0],
             "sigma0": [round(float(v), 6) for v in self.sigma0], "coupled": self.coupled}
        if self.c_max is not None:
            d["c_max"] = [round(float(v), 6) for v in self.c_max]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, indent=1)
            f.write("\n")

    def reorder(self, names: list[str]) -> "Priors":
        """The same priors in another expression order (missing names -> identity values)."""
        ix = {n: i for i, n in enumerate(self.expressions)}
        pick = lambda a, dflt: np.array([a[ix[n]] if n in ix else dflt for n in names], float)
        return Priors(list(names), pick(self.c0, 1.0), pick(self.b0, 0.0), pick(self.sigma0, 0.005),
                      self.coupled, self.meta, pick(self.c_max, 0.0) if self.c_max is not None else None)


def soft_knee(u: np.ndarray, d: np.ndarray) -> np.ndarray:
    """C1 deadzone: 0 below 0, quadratic on (0, 2d), u - d above."""
    d = np.maximum(d, 1e-9)
    return np.where(u <= 0, 0.0, np.where(u < 2 * d, u * u / (4 * d), u - d))


class AdaptiveCalibrator:
    def __init__(self, priors: Priors, params: Params | None = None):
        self.p = p = params or Params()
        self.pr = priors
        self.names = priors.expressions
        n = self.n = len(self.names)
        self.c0 = np.clip(priors.c0, p.c_abs_min, 1.0)
        self.c_lo = np.maximum(self.c0 / p.bound_down, p.c_abs_min)
        c_max = priors.c_max if priors.c_max is not None else np.zeros(n)
        self.c_hi = np.minimum(np.maximum(self.c0 * p.bound_up, c_max), 1.0)

        # learned state (persisted)
        self.active_s = 0.0
        self.c = self.c0.copy()
        self.g = np.ones(n)
        self.b = priors.b0.copy()
        self.sigma = np.clip(priors.sigma0, p.sigma_floor, p.sigma_max)
        self.hist = np.zeros((n, p.hist_bins))
        self.hist_n = 0.0  # decayed sample count (same for every shape)
        self.peak_v = np.zeros((n, p.peaks_keep))
        self.peak_t = np.zeros((n, p.peaks_keep))
        self.peak_n = np.zeros(n, dtype=int)

        # runtime state (not persisted)
        self._prev_x = None
        self._prev_t = None
        self._last_change_t = None
        self._pause_until = -math.inf
        self._was_bad = False
        self._ring_t = np.full(64, -math.inf)
        self._ring_x = np.zeros((64, n))
        self._ring_i = 0
        self._hist_acc = 0.0
        self._update_acc = 0.0
        self._ep_on = np.zeros(n, dtype=bool)
        self._ep_t0 = np.zeros(n)
        self._ep_peak = np.zeros(n)
        self._ep_bad = np.zeros(n, dtype=bool)
        self._sat_run = np.zeros(n)  # current pinned run length, active seconds
        self._sat_short = np.zeros(n)  # decayed seconds spent in short pinned runs
        self._above = np.zeros(n)  # decayed seconds busy (shown >= guard_busy_level, or above theta)
        self._sat_new = np.zeros(n)  # seconds in short pinned runs since this shape's last guardrail step
        # active minute of the latest short pin per slot (slot = minute % guard_minutes); -1e9 = none
        self._pin_min = np.full((n, max(p.guard_minutes, 1)), -1e9)
        self._guard_acc = 0.0
        self.gated = False
        self.h = np.zeros(n)

        # coupled shapes: child index -> (parent index, ratio)
        ix = {nm: i for i, nm in enumerate(self.names)}
        self.coupled = {ix[ch]: (ix[v["parent"]], float(v["ratio"]))
                        for ch, v in priors.coupled.items() if ch in ix and v["parent"] in ix}

    # ------------------------------------------------------------------ frame

    def step(self, t: float, x_in: np.ndarray) -> np.ndarray:
        p = self.p
        x = np.clip(np.nan_to_num(np.asarray(x_in, float), nan=0.0), 0.0, 1.0)
        nan_frame = bool(np.isnan(np.asarray(x_in, float)).any())

        # 6.1 gating
        if self._prev_x is None:
            self._prev_x, self._prev_t, self._last_change_t = x.copy(), t, t
        dx = np.abs(x - self._prev_x)
        if dx.max() > p.frozen_eps:
            self._last_change_t = t
        zeros = not x.any()
        frozen = (t - self._last_change_t) >= p.frozen_s
        if int((dx > p.jump_size).sum()) >= p.jump_count:
            self._pause_until = max(self._pause_until, t + p.jump_pause_s)
        bad = zeros or frozen or nan_frame
        if self._was_bad and not bad:
            self._pause_until = max(self._pause_until, t + p.recover_s)
        self._was_bad = bad
        self.gated = gated = bad or t < self._pause_until
        dt = min(max(t - self._prev_t, 0.0), p.max_frame_dt)
        self._prev_x, self._prev_t = x, t
        adt = 0.0 if gated else dt
        self.active_s += adt

        # held signal (rolling min over hold_s)
        self._ring_t[self._ring_i] = t
        self._ring_x[self._ring_i] = x
        self._ring_i = (self._ring_i + 1) % len(self._ring_t)
        self.h = h = self._ring_x[self._ring_t >= t - p.hold_s].min(axis=0)

        if not gated:
            # 6.2 rest histogram, 20 Hz on the active clock
            self._hist_acc += adt
            if self._hist_acc >= 1.0 / p.hist_hz:
                self._hist_acc -= 1.0 / p.hist_hz
                decay = 2.0 ** (-1.0 / (p.hist_hz * p.hist_half_life_s))
                self.hist *= decay
                self.hist_n = self.hist_n * decay + 1.0
                bins = np.minimum((x * p.hist_bins).astype(int), p.hist_bins - 1)
                self.hist[np.arange(self.n), bins] += 1.0

        # 6.3 episodes on the held signal
        theta = self.b + p.theta_k * self.sigma + p.theta_add
        above = h > theta
        start = above & ~self._ep_on
        self._ep_t0[start] = t
        self._ep_peak[start] = h[start]
        self._ep_bad[start] = gated
        cont = above & self._ep_on
        self._ep_peak[cont] = np.maximum(self._ep_peak[cont], h[cont])
        if gated:
            self._ep_bad[cont] = True
        ended = np.nonzero(self._ep_on & ~above)[0]
        for i in ended:
            if not self._ep_bad[i] and (t - self._ep_t0[i]) >= p.episode_min_s:
                self._add_peak(i, self._ep_peak[i])
        self._ep_on = above

        # output with the current state
        y = self._map(x)

        # 6.4 guardrail bookkeeping
        if p.guardrail and not gated:
            w = 2.0 ** (-adt / p.guard_window_s)
            pinned = y >= p.sat_level
            run_end = (~pinned) & (self._sat_run > 0)
            short = run_end & (self._sat_run < p.sat_run_max_s)
            self._sat_short = self._sat_short * w + np.where(short, self._sat_run, 0.0)
            self._sat_new = self._sat_new + np.where(short, self._sat_run, 0.0)
            if short.any():
                minute = math.floor(self.active_s / 60.0)
                self._pin_min[short, minute % self._pin_min.shape[1]] = minute
            self._sat_run = np.where(pinned, self._sat_run + adt, 0.0)
            busy = above if p.guard_busy_level <= 0 else (y >= p.guard_busy_level)
            self._above = self._above * w + np.where(busy, adt, 0.0)

        # periodic learning
        if not gated:
            self._update_acc += adt
            if self._update_acc >= p.update_every_s:
                self._update(self._update_acc)
                self._update_acc = 0.0
        return y

    def _map(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        d = p.deadzone_k * self.sigma
        z = soft_knee(x - self.b, d)
        den = np.maximum(self.c * self.g - self.b - d, p.min_denominator)
        y = np.clip(z / den, 0.0, 1.0)
        for ch, (par, ratio) in self.coupled.items():
            y[ch] = min(max(x[ch] * ratio / den[par], 0.0), 1.0)
        return y

    # ------------------------------------------------------------------ learning

    def _add_peak(self, i: int, v: float):
        k = self.peak_n[i]
        if k < self.p.peaks_keep:
            self.peak_v[i, k], self.peak_t[i, k] = v, self.active_s
            self.peak_n[i] = k + 1
            return
        w = 2.0 ** (-(self.active_s - self.peak_t[i]) / self.p.peak_half_life_s)
        j = int(np.argmin(self.peak_v[i] * w))
        if v > self.peak_v[i, j] * w[j]:
            self.peak_v[i, j], self.peak_t[i, j] = v, self.active_s

    def _update(self, elapsed: float):
        p = self.p
        # 6.2 rest + noise from the histogram
        if self.hist_n >= p.rest_min_samples:
            q05, q25, q50 = self._quantiles((p.rest_q, 0.25, 0.5))
            ok = (q50 - q05) <= p.rest_max_spread
            sig = self._lower_half_mad(q25, q50) * 1.4826
            self.b = np.minimum(np.where(ok, q05, self.pr.b0), self.pr.b0 + p.rest_max_rise)
            self.sigma = np.clip(np.where(ok, sig, self.pr.sigma0), p.sigma_floor, p.sigma_max)

        # 6.3 ceilings
        # log(1 + x), not log1p: .NET has no true log1p, and the C# port must match bit for bit (golden_v0.py)
        slew_up = math.log(1.0 + p.slew_up_per_min) * elapsed / 60.0
        slew_down = math.log(1.0 + p.slew_per_10min) * elapsed / 600.0
        for i in np.nonzero(self.peak_n)[0]:
            k = self.peak_n[i]
            v, ts = self.peak_v[i, :k], self.peak_t[i, :k]
            w = 2.0 ** (-(self.active_s - ts) / p.peak_half_life_s)
            order = np.argsort(-v)
            cum = np.cumsum(w[order])
            if cum[-1] < p.k_weight:
                continue  # fewer than ~3 recent held peaks: no evidence either way
            c_obs = v[order][int(np.searchsorted(cum, p.k_weight))]
            if c_obs > self.c[i]:
                target = c_obs  # held expressions go past the ceiling: follow them (calmer, never pinned)
            else:
                n_eff = min(float(cum[-1]), p.n_eff_max)
                log_c = (p.n0 * math.log(self.c0[i]) + n_eff * math.log(p.eta * c_obs)) / (p.n0 + n_eff)
                target = math.exp(log_c)
            target = min(max(target, self.c_lo[i]), self.c_hi[i])
            step = min(max(math.log(target / self.c[i]), -slew_down), slew_up)
            self.c[i] = min(max(self.c[i] * math.exp(step), self.c_lo[i]), self.c_hi[i])

        # 6.4 guardrail
        if p.guardrail:
            self._guard_acc += elapsed
            relax = 2.0 ** (-elapsed / p.guard_half_life_s)
            self.g = 1.0 + (self.g - 1.0) * relax
            if self._guard_acc >= p.guard_every_s:
                self._guard_acc = 0.0
                share = self._sat_short / np.maximum(self._above, 1e-9)
                hot = (share > p.guard_share) & (self._above > 10.0)
                if p.guard_needs_new_pins:
                    hot &= self._sat_new > 0.0
                if p.guard_min_minutes > 0:
                    # slots holding one of the last guard_minutes active minutes = minutes with short pins
                    recent = self._pin_min > math.floor(self.active_s / 60.0) - p.guard_minutes
                    hot &= recent.sum(axis=1) >= p.guard_min_minutes
                self.g = np.where(hot, self.g * p.guard_step, self.g)
                self._sat_new = np.where(hot, 0.0, self._sat_new)
                self.g = np.minimum(self.g, self.c_hi / self.c)

    def _quantiles(self, qs):
        cdf = np.cumsum(self.hist, axis=1)
        tot = cdf[:, -1:]
        width = 1.0 / self.p.hist_bins
        out = []
        for q in qs:
            k = np.argmax(cdf >= q * tot, axis=1)
            out.append((k + 0.5) * width)
        return out

    def _lower_half_mad(self, q25: np.ndarray, q50: np.ndarray) -> np.ndarray:
        width = 1.0 / self.p.hist_bins
        centers = (np.arange(self.p.hist_bins) + 0.5) * width
        res = np.empty(self.n)
        for i in range(self.n):
            m = centers <= q50[i] + 1e-12
            dist, wts = np.abs(centers[m] - q25[i]), self.hist[i, m]
            if wts.sum() <= 0:
                res[i] = 0.0
                continue
            o = np.argsort(dist)
            cw = np.cumsum(wts[o])
            res[i] = dist[o][np.searchsorted(cw, 0.5 * cw[-1])]
        return res

    # ------------------------------------------------------------------ state

    def state(self) -> dict:
        return {"schema": 1, "params": asdict(self.p), "active_s": self.active_s,
                "c": self.c.tolist(), "g": self.g.tolist(), "b": self.b.tolist(), "sigma": self.sigma.tolist(),
                "hist_n": self.hist_n, "hist": self.hist.tolist(), "peak_v": self.peak_v.tolist(),
                "peak_t": self.peak_t.tolist(), "peak_n": self.peak_n.tolist()}

    def load_state(self, s: dict):
        self.active_s = s["active_s"]
        for k in ("c", "g", "b", "sigma", "hist", "peak_v", "peak_t"):
            setattr(self, k, np.array(s[k], float))
        self.hist_n = s["hist_n"]
        self.peak_n = np.array(s["peak_n"], int)
        # state saved under older limits (e.g. before sigma_max) follows the current ones from the first frame
        self.b = np.minimum(self.b, self.pr.b0 + self.p.rest_max_rise)
        self.sigma = np.clip(self.sigma, self.p.sigma_floor, self.p.sigma_max)
