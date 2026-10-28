# Cloud Workload Forecasting (Atomity AI/ML take-home)

Forecasts `active_vm_count` and `aggregate_cpu_utilization` for the final two hours
(24 five-minute steps) of the Atomity `data-v1` release (Azure Public Dataset V2 subset).
The full write-up is in [`reports/final_report.md`](reports/final_report.md).

## Install (Python 3.11+)

Linux / macOS:

    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -e ".[dev]"

Windows PowerShell:

    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"

Run the tests: `python -m pytest -q`

## Data

Only the official `data-v1` release is used (no other dataset or version); it is not committed.
Download `data-v1.tar.gz` from https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
and extract it into `challenge_data/`:

    mkdir challenge_data
    tar -xzf data-v1.tar.gz -C challenge_data

Optional integrity check against the release's `SHA256SUMS`: add `--verify-release` to the command below.
The pipeline itself only needs `vm_cpu_readings.csv.gz`, so smaller smoke datasets also run.

## Generate the predictions (one command)

    python -m cwf.predict --data challenge_data --output predictions.csv --report-dir reports/tables

## Reproduce the analysis

    # Linux/macOS:  CWF_DATA=challenge_data python notebooks/eda.py
    # PowerShell:   $env:CWF_DATA="challenge_data"; python notebooks/eda.py
    python scripts/policy_report.py --data challenge_data

`notebooks/eda.ipynb` contains the same cells as `notebooks/eda.py`
(regenerate with `python notebooks/make_ipynb.py`). Figures go to `reports/figures/`, tables to `reports/tables/`.

## Main decisions

- Targets are built from the full series first, then split: first n-24 timestamps train, last 24 holdout.
- Holdout protocol: the final 24 timestamps only label the output rows. Their target values are never
  used for fitting, feature selection, tuning or model selection. All validation is rolling-origin
  inside the first 110 timestamps (origins 50, 62, 74, 86; horizon 24).
- Valid reading = non-null `avg_cpu` in [0, 100]; duplicate (timestamp, vm_id) rows dropped;
  each VM counted once per timestamp. Timestamp gaps are reported (warning), never imputed.
- Candidates: naive, 6/12-step moving average, drift, SES, damped Holt. Selection by backtest MAE
  per target; ties go to the simpler model; series too short for a backtest fall back to naive.
- Result: naive was selected for both targets (see the report for the evidence and its limits).

## Layout

    src/cwf/        data.py, models.py, backtest.py, policy.py, predict.py
    tests/          submission-contract and smoke tests
    notebooks/      eda.py, eda.ipynb, make_ipynb.py
    scripts/        policy_report.py (error analysis + headroom simulation)
    reports/        final_report.md, figures/, tables/
## Completed Results

- **Forecast output:** `predictions.csv` in the project root.
- **Forecast horizon:** Final two hours, comprising 24 five-minute intervals.
- **Forecast targets:** `active_vm_count` and `aggregate_cpu_utilization`.
- **Automated tests:** 9 passed.
- **Selected models:** Naive forecasting for both targets.
- **Data cleaning:** 1,340,000 rows processed; 0 duplicates and 0 invalid rows reported.
- **Final analysis:** `reports/final_report.md`

### Reproduce the Forecast

```bash
python -m cwf.predict --data challenge_data --output predictions.csv --report-dir reports/tables
```

### Screenshots

<img width="605" height="377" alt="Screenshot 2026-10-28 230329" src="https://github.com/user-attachments/assets/56b4bdbb-435b-40f3-90e9-84c313423baf" />



<img width="421" height="379" alt="Screenshot 2026-10-28 231527" src="https://github.com/user-attachments/assets/6de58ebe-c760-4a80-8452-e13b92cae73e" />



<img width="503" height="224" alt="Screenshot 2026-10-28 231826" src="https://github.com/user-attachments/assets/577fdda3-0d43-4ffe-a97f-324d946c8bf4" />



