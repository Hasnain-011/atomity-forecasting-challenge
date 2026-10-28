"""Rolling-origin evaluation. Only ever pass TRAINING data in here."""
import numpy as np
import pandas as pd

from .models import CANDIDATES


def rolling_origins(n, horizon, min_train, step):
    out, o = [], n - horizon
    while o >= min_train:
        out.append(o)
        o -= step
    return sorted(out)


def backtest(y, forecaster, horizon=24, min_train=48, step=12):
    y = np.asarray(y, dtype=float)
    rows = []
    for o in rolling_origins(len(y), horizon, min_train, step):
        pred = np.asarray(forecaster(y[:o], horizon))
        for h in range(horizon):
            rows.append((o, h + 1, y[o + h], pred[h]))
    return pd.DataFrame(rows, columns=["origin", "h", "y", "yhat"])


def metrics(bt, peak_quantile=0.75):
    err = bt["y"] - bt["yhat"]
    thr = bt["y"].quantile(peak_quantile)
    peak = bt["y"] >= thr
    return {"mae": err.abs().mean(), "rmse": float(np.sqrt((err ** 2).mean())),
            "peak_mae": err[peak].abs().mean(), "n": len(bt)}


def compare(y, horizon=24, min_train=48, step=12):
    """Backtest every candidate. Returns (table, dict of backtest frames)."""
    frames, rows = {}, []
    for name, f in CANDIDATES.items():
        bt = backtest(y, f, horizon, min_train, step)
        if bt.empty:
            continue
        frames[name] = bt
        rows.append({"model": name, **metrics(bt)})
    if not rows:
        return pd.DataFrame(), frames
    tab = pd.DataFrame(rows).set_index("model")
    tab["skill_vs_naive"] = 1 - tab["mae"] / tab.loc["naive", "mae"]
    return tab, frames


def select_model(y, horizon=24, min_train=48, step=12):
    """Pick lowest backtest MAE; fall back to naive if series too short."""
    tab, frames = compare(y, horizon, min_train, step)
    if tab.empty:
        return "naive", tab, frames
    best = tab["mae"].idxmin()  # idxmin returns first min -> simpler wins ties
    return best, tab, frames
