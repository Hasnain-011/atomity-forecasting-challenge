"""Small, dependency-free forecasters. Each: f(y: np.ndarray, h: int) -> np.ndarray[h]."""
import itertools
import numpy as np


def naive(y, h):
    return np.repeat(y[-1], h)


def mean_k(k):
    def f(y, h):
        return np.repeat(y[-k:].mean(), h)
    f.__name__ = f"mean_{k}"
    return f


def drift(w):
    def f(y, h):
        w_ = min(w, len(y))
        slope = (y[-1] - y[-w_]) / (w_ - 1) if w_ > 1 else 0.0
        return y[-1] + slope * np.arange(1, h + 1)
    f.__name__ = f"drift_{w}"
    return f


def _holt(y, a, b, phi):
    """Damped Holt filter. Returns one-step SSE, final level, final trend."""
    l, t, sse = y[0], 0.0, 0.0
    for x in y[1:]:
        pred = l + phi * t
        sse += (x - pred) ** 2
        l_new = a * x + (1 - a) * pred
        t = b * (l_new - l) + (1 - b) * phi * t
        l = l_new
    return sse, l, t


def ses(y, h):
    best = min(((_holt(y, a, 0.0, 0.0)[0], a) for a in np.linspace(0.05, 1, 20)))
    _, l, _ = _holt(y, best[1], 0.0, 0.0)
    return np.repeat(l, h)


def holt_damped(y, h):
    grid = itertools.product([0.1, 0.3, 0.5, 0.8], [0.05, 0.1, 0.3], [0.8, 0.9, 0.98])
    _, (a, b, phi) = min((_holt(y, a, b, p)[0], (a, b, p)) for a, b, p in grid)
    _, l, t = _holt(y, a, b, phi)
    steps = np.arange(1, h + 1)
    return l + t * np.cumsum(phi ** steps)


# Order matters: earlier = simpler; ties go to the simpler model.
CANDIDATES = {
    "naive": naive,
    "mean_6": mean_k(6),
    "mean_12": mean_k(12),
    "drift_12": drift(12),
    "ses": ses,
    "holt_damped": holt_damped,
}
