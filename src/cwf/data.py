"""Load, validate and aggregate an Atomity ``data-v1`` release."""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pandas as pd

HOLDOUT = 24
REQUIRED = ["timestamp", "vm_id", "avg_cpu"]
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]
RELEASE_FILES = ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz", "DATA_CARD.md", "SHA256SUMS")


def verify_release(data_dir):
    """Verify the files declared in ``SHA256SUMS`` and return their paths.

    Optional (``--verify-release``): the pipeline itself only needs
    ``vm_cpu_readings.csv.gz`` so that smoke datasets without the other release
    files still run.
    """
    root = Path(data_dir)
    absent = [name for name in RELEASE_FILES if not (root / name).is_file()]
    if absent:
        raise FileNotFoundError(f"data release is missing: {', '.join(absent)}")
    expected = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and len(fields[0]) == 64:
            expected[fields[1].lstrip(" *")] = fields[0].lower()
    if not expected:
        raise ValueError("SHA256SUMS contains no SHA-256 entries")
    for name in ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz", "DATA_CARD.md"):
        if name not in expected:
            raise ValueError(f"SHA256SUMS has no entry for {name}")
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected[name]:
            raise ValueError(f"SHA-256 mismatch for {name}")
    return root


def load_readings(data_dir):
    df = pd.read_csv(Path(data_dir) / "vm_cpu_readings.csv.gz")
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"vm_cpu_readings is missing columns: {sorted(missing)}")
    return df


def clean(df):
    """Valid reading = non-null avg_cpu within [0, 100]; one row per (timestamp, vm_id)."""
    n0 = len(df)
    df = df.drop_duplicates(["timestamp", "vm_id"])
    valid = df["avg_cpu"].between(0, 100)  # NaN -> False
    report = {"rows": n0, "duplicates": n0 - len(df), "invalid": int((~valid).sum())}
    return df[valid], report


def build_targets(df):
    g = df.groupby("timestamp")
    out = pd.DataFrame({
        "active_vm_count": g["vm_id"].nunique().astype(float),
        "aggregate_cpu_utilization": g["avg_cpu"].mean(),
    })
    return out.sort_index()


def split(targets, holdout=HOLDOUT):
    """Chronological split returning training targets and holdout *timestamps*.

    Selecting the index is safe; the final target values are never read by the
    forecasting path.
    """
    if len(targets) <= holdout:
        raise ValueError(f"need more than {holdout} timestamps, got {len(targets)}")
    return targets.iloc[:-holdout], targets.index[-holdout:].copy()


def gap_report(index):
    steps = pd.Series(index).diff().dropna()
    return {"n_timestamps": len(index),
            "step_counts": steps.value_counts().to_dict()}


def validate_five_minute_index(index, strict=True):
    """Reject duplicate/unordered timestamps; irregular steps raise if strict, else warn.

    Gaps are reported, never imputed (no target values are invented).
    """
    index = pd.Index(index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("aggregate timestamps must be unique and chronological")
    if len(index) < 2:
        return
    steps = pd.Series(index).diff().dropna()
    expected = pd.Timedelta(minutes=5) if isinstance(index, pd.DatetimeIndex) else 300
    if not (steps == expected).all():
        msg = "aggregate timestamps are not a regular five-minute grid; gaps are not imputed"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning)
