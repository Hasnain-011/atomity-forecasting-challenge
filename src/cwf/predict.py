"""python -m cwf.predict --data challenge_data --output predictions.csv"""
import argparse
import json
import numpy as np
import pandas as pd

from . import data as D
from .backtest import select_model
from .models import CANDIDATES


def validate_output(out, holdout_timestamps, holdout=D.HOLDOUT):
    """Enforce the submission schema before any CSV is written."""
    expected = ["timestamp", *D.TARGETS]
    if list(out.columns) != expected or len(out) != holdout:
        raise ValueError("predictions must have 24 rows and the required columns")
    if out["timestamp"].duplicated().any() or not out["timestamp"].tolist() == list(holdout_timestamps):
        raise ValueError("prediction timestamps must exactly match the chronological holdout")
    values = out[D.TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    if (out["active_vm_count"] < 0).any() or not out["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("predictions are outside their allowed ranges")


def run(data_dir, holdout=D.HOLDOUT, min_train=48, step=12, report_dir=None, verify_release=False):
    if verify_release:
        D.verify_release(data_dir)
    raw = D.load_readings(data_dir)
    df, rep = D.clean(raw)
    targets = D.build_targets(df)
    D.validate_five_minute_index(targets.index, strict=False)
    train, holdout_timestamps = D.split(targets, holdout)  # holdout VALUES are never read below
    out = pd.DataFrame({"timestamp": holdout_timestamps.to_numpy()})
    chosen = {}
    for col in D.TARGETS:
        y = train[col].to_numpy()
        name, tab, _ = select_model(y, holdout, min_train, step)
        chosen[col] = name
        pred = CANDIDATES[name](y, holdout)  # refit on the full training window
        if col == "active_vm_count":
            pred = np.clip(pred, 0, None)
        else:
            pred = np.clip(pred, 0, 100)
        assert np.isfinite(pred).all()
        out[col] = pred
        if report_dir is not None and not tab.empty:
            import os
            os.makedirs(report_dir, exist_ok=True)
            tab.to_csv(f"{report_dir}/backtest_{col}.csv")
    out = out[["timestamp", *D.TARGETS]]
    validate_output(out, holdout_timestamps, holdout)
    return out, {"cleaning": rep, "chosen": chosen}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--holdout", type=int, default=D.HOLDOUT)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--verify-release", action="store_true",
                   help="optionally verify the Atomity data-v1 release against SHA256SUMS first")
    p.add_argument("--skip-release-verification", action="store_true",
                   help=argparse.SUPPRESS)  # deprecated no-op, kept for compatibility
    a = p.parse_args()
    out, info = run(a.data, a.holdout, report_dir=a.report_dir, verify_release=a.verify_release)
    out.to_csv(a.output, index=False)
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
