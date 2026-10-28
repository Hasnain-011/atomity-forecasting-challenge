"""Error-by-horizon, error-by-origin, peak error and headroom-policy simulation.

Uses ONLY the training window (first n-24 timestamps); holdout values are never touched.
    python scripts/policy_report.py --data challenge_data
"""
import argparse
import os

import numpy as np
import pandas as pd

from cwf import data as D
from cwf import policy as P
from cwf.backtest import select_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--out", default="reports/tables")
    ap.add_argument("--min-train", type=int, default=48)
    ap.add_argument("--step", type=int, default=12)
    a = ap.parse_args()

    df, _ = D.clean(D.load_readings(a.data))
    train, _ = D.split(D.build_targets(df))  # holdout values are never used
    col = "aggregate_cpu_utilization"
    y = train[col].to_numpy()
    name, tab, frames = select_model(y, D.HOLDOUT, a.min_train, a.step)
    if name not in frames:
        print("Series too short for a backtest; nothing to report.")
        return
    bt = frames[name].copy()
    bt["err"] = bt["y"] - bt["yhat"]
    bt["abs_err"] = bt["err"].abs()
    os.makedirs(a.out, exist_ok=True)

    horizon = bt.groupby("h").agg(mae=("abs_err", "mean"),
                                  rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))),
                                  bias=("err", "mean"))
    origin = bt.groupby("origin").agg(mae=("abs_err", "mean"), bias=("err", "mean"))
    origin["window_start_step"] = origin.index
    origin["window_end_step"] = origin.index + D.HOLDOUT - 1
    thr = bt["y"].quantile(0.75)
    peak = bt["y"] >= thr
    peak_tab = pd.DataFrame({"mae": [bt.loc[peak, "abs_err"].mean(), bt.loc[~peak, "abs_err"].mean()],
                             "bias": [bt.loc[peak, "err"].mean(), bt.loc[~peak, "err"].mean()],
                             "n": [int(peak.sum()), int((~peak).sum())]},
                            index=["peak (top quartile)", "non-peak"])
    pol = P.simulate(bt)
    level = float(y.mean())
    pol["margin_pct_of_mean_level"] = pol["margin"] / level * 100

    horizon.to_csv(f"{a.out}/error_by_horizon.csv")
    origin.to_csv(f"{a.out}/error_by_origin.csv")
    peak_tab.to_csv(f"{a.out}/error_peak_vs_nonpeak.csv")
    pol.to_csv(f"{a.out}/headroom_policy.csv")

    print(f"model used: {name}; mean training level: {level:.3f}%")
    print(f"overall MAE as % of mean level: {bt['abs_err'].mean() / level * 100:.2f}%")
    print("\nError by horizon step (selected rows):\n", horizon.loc[[1, 6, 12, 18, 24]].round(3).to_string())
    print("\nError by origin (window = steps origin..origin+23):\n", origin.round(3).to_string())
    print("\nPeak vs non-peak (threshold %.3f):\n" % thr, peak_tab.round(3).to_string())
    print("\nHeadroom policy (margin in percentage points above forecast):\n", pol.round(3).to_string())
    print(f"\nsaved CSVs to {a.out}")


if __name__ == "__main__":
    main()
