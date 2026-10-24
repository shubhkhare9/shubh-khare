"""Small robust-statistics helpers shared by the checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def mad_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def runs_of(mask: pd.Series) -> list[tuple[int, int]]:
    """Positions (start, end inclusive) of consecutive True runs in a boolean Series."""
    m = np.asarray(mask.fillna(False).values, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by at most ``max_gap`` samples."""
    out: list[list[int]] = []
    for s, e in runs:
        if out and s - out[-1][1] - 1 <= max_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def hampel_deviation(x: pd.Series, half_window: int = 3, min_scale: float = 0.0):
    """Deviation of each point from the rolling median, and the local robust scale.

    A point that is far from the median of its +-half_window neighbours (and the neighbours
    themselves agree) is a spike; a genuine weather change moves the neighbours too.
    """
    w = 2 * half_window + 1
    med = x.rolling(w, center=True, min_periods=half_window + 1).median()
    dev = x - med
    scale = (x - med).abs().rolling(w, center=True, min_periods=half_window + 1).median() * MAD_TO_SIGMA
    return dev, scale.clip(lower=min_scale)


def best_split(y: np.ndarray, min_seg: int):
    """Least-squares single change point of a mean level. Returns (index, delta, score) or None.

    ``index`` is the first sample of the right-hand segment. NaNs are ignored (handled by
    the caller passing compact arrays).
    """
    n = len(y)
    if n < 2 * min_seg:
        return None
    c = np.cumsum(y)
    k = np.arange(min_seg, n - min_seg + 1)
    left_mean = c[k - 1] / k
    right_mean = (c[-1] - c[k - 1]) / (n - k)
    score = np.abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n)
    j = int(np.argmax(score))
    return int(k[j]), float(right_mean[j] - left_mean[j]), float(score[j])


def segment_levels(y: pd.Series, min_seg: int, min_delta: float, noise_mult: float = 4.0, max_splits: int = 6):
    """Binary segmentation of a (NaN-tolerant) series into constant-level segments.

    A split is accepted when the level changes by at least ``min_delta`` AND by at least
    ``noise_mult`` times the robust noise of the series around its segment means. Returns a list of
    dicts: start/end positions (in the *original* index), mean level, n.
    """
    valid = y.dropna()
    if len(valid) < 2 * min_seg:
        return []
    pos = np.where(~y.isna().values)[0]
    vals = valid.values
    segs = [(0, len(vals))]       # in compact coordinates
    accepted: list[tuple[int, int]] = []
    queue = [(0, len(vals))]
    while queue and len(accepted) < max_splits + 1:
        a, b = queue.pop()
        sp = best_split(vals[a:b], min_seg)
        if sp is None:
            accepted.append((a, b)); continue
        k, delta, _ = sp
        left, right = vals[a:a + k], vals[a + k:b]
        resid = np.concatenate([left - np.median(left), right - np.median(right)])
        noise = mad_sigma(resid)
        noise = max(noise, 1e-9)
        if abs(delta) >= min_delta and abs(delta) >= noise_mult * noise:
            queue += [(a, a + k), (a + k, b)]
        else:
            accepted.append((a, b))
    accepted.sort()
    out = []
    for a, b in accepted:
        out.append({"start_pos": int(pos[a]), "end_pos": int(pos[b - 1]),
                    "level": float(np.median(vals[a:b])), "n": int(b - a)})
    return out


def hinge_fit(y: pd.Series, min_side: int = 7):
    """Fit a flat level that starts to ramp linearly at an unknown day (a "hinge").

    Returns dict(onset, base, rise, sigma, gain) or None. ``rise`` is the total change from the onset to
    the last sample, ``gain`` the share of the variance explained compared with a constant (0..1) and
    ``sigma`` the scatter of the day-to-day residuals. A gradual drift has a high gain and rise >> sigma;
    healthy stations have gain below ~0.2 in the supplied pack.
    """
    v = y.dropna()
    if len(v) < 2 * min_side + 2:
        return None
    k = np.array([(d - v.index[0]).days for d in v.index], float)
    n = len(v)
    sse0 = float(((v.values - v.values.mean()) ** 2).sum())
    if sse0 <= 0:
        return None
    best = None
    for k0 in range(min_side, int(k.max()) - min_side + 1):
        h = np.clip(k - k0, 0, None)
        X = np.c_[np.ones(n), h]
        beta, *_ = np.linalg.lstsq(X, v.values, rcond=None)
        sse = float(((v.values - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k0, float(beta[0]), float(beta[1]), int(k.max()) - k0)
    sse, k0, a, b, m = best
    return {"onset": v.index[0] + pd.Timedelta(days=k0), "base": a, "rise": b * m,
            "sigma": float(np.sqrt(sse / max(n - 3, 1))), "gain": 1 - sse / sse0}
