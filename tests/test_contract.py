import gzip
import hashlib
import numpy as np
import pandas as pd
import pytest

from cwf.data import verify_release
from cwf.predict import run, validate_output


def make_data(tmp_path, n_ts=134, n_vm=60, churn=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vm):
        start = rng.integers(0, 10) if churn else 0
        base = rng.uniform(5, 60)
        for t in range(start, n_ts):
            rows.append((t * 300, f"vm{v}", 0, 100, float(np.clip(base + rng.normal(0, 5), 0, 100))))
    df = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    df = pd.concat([df, df.iloc[:5]])  # duplicates
    df.loc[df.index[10], "avg_cpu"] = np.nan
    df.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    return tmp_path


def check_contract(out, expected_ts):
    assert list(out.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(out) == 24
    assert out["timestamp"].tolist() == expected_ts
    assert not out["timestamp"].duplicated().any()
    assert np.isfinite(out[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()).all()
    assert (out["active_vm_count"] >= 0).all()
    assert out["aggregate_cpu_utilization"].between(0, 100).all()


def test_full_size(tmp_path):
    d = make_data(tmp_path)
    out, info = run(str(d))
    check_contract(out, [t * 300 for t in range(110, 134)])


def test_tiny_dataset_falls_back(tmp_path):
    d = make_data(tmp_path, n_ts=40, n_vm=5)
    out, info = run(str(d))
    check_contract(out, [t * 300 for t in range(16, 40)])
    assert set(info["chosen"].values()) == {"naive"}


def test_holdout_values_not_used(tmp_path):
    d = make_data(tmp_path, churn=False)
    out1, _ = run(str(d))
    df = pd.read_csv(d / "vm_cpu_readings.csv.gz")
    df.loc[df.timestamp >= 110 * 300, "avg_cpu"] = 99.0  # corrupt holdout
    df.to_csv(d / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    out2, _ = run(str(d))
    pd.testing.assert_frame_equal(out1, out2)


def test_output_rejects_wrong_holdout_timestamp(tmp_path):
    d = make_data(tmp_path)
    out, _ = run(str(d))
    out.loc[0, "timestamp"] = -1
    with pytest.raises(ValueError, match="timestamps"):
        validate_output(out, [t * 300 for t in range(110, 134)])


def test_non_five_minute_timestamps_warn_and_return_forecast(tmp_path):
    d = make_data(tmp_path)
    frame = pd.read_csv(d / "vm_cpu_readings.csv.gz")
    frame.loc[frame.timestamp == 300, "timestamp"] = 301
    frame.to_csv(d / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    with pytest.warns(UserWarning, match="five-minute"):
        out, _ = run(str(d))
    assert len(out) == 24


def test_release_checksum_is_enforced(tmp_path):
    d = make_data(tmp_path)
    (d / "vm_metadata.csv.gz").write_bytes(b"metadata")
    (d / "DATA_CARD.md").write_text("synthetic release", encoding="utf-8")
    readings = (d / "vm_cpu_readings.csv.gz").read_bytes()
    metadata = (d / "vm_metadata.csv.gz").read_bytes()
    card = (d / "DATA_CARD.md").read_bytes()
    (d / "SHA256SUMS").write_text(
        f"{hashlib.sha256(readings).hexdigest()}  vm_cpu_readings.csv.gz\n"
        f"{hashlib.sha256(metadata).hexdigest()}  vm_metadata.csv.gz\n"
        f"{hashlib.sha256(card).hexdigest()}  DATA_CARD.md\n",
        encoding="utf-8",
    )
    verify_release(d)
    (d / "vm_metadata.csv.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_release(d)
