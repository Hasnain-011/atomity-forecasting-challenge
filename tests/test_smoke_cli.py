import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cwf.data import validate_five_minute_index
from cwf.predict import run


def write_readings(path, n_ts=40, n_vm=5, drop=None):
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n_ts):
        if drop is not None and t == drop:
            continue
        for v in range(n_vm):
            rows.append((t * 300, f"vm{v}", 0.0, 100.0, float(rng.uniform(1, 60))))
    pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]).to_csv(
        path / "vm_cpu_readings.csv.gz", index=False)


def test_cli_needs_only_the_readings_file(tmp_path):
    write_readings(tmp_path)  # no SHA256SUMS, no metadata, no DATA_CARD
    out = tmp_path / "p.csv"
    subprocess.run([sys.executable, "-m", "cwf.predict", "--data", str(tmp_path),
                    "--output", str(out)], check=True)
    p = pd.read_csv(out)
    assert list(p.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(p) == 24


def test_irregular_grid_warns_instead_of_crashing(tmp_path):
    write_readings(tmp_path, drop=10)
    with pytest.warns(UserWarning):
        out, _ = run(str(tmp_path))
    assert len(out) == 24


def test_strict_validation_still_rejects_irregular_steps():
    with pytest.raises(ValueError):
        validate_five_minute_index([0, 300, 900])
