# Cloud Workload Forecasting and Analysis: Final Report

*Hasnain Ali.*

## 1. Executive summary

The workload in the `data-v1` release is very predictable at a two-hour horizon, but mostly because it is
almost static. All 10,000 VMs report at every one of the 110 training timestamps, so `active_vm_count` is
constant at 10,000 and the correct forecast is 10,000. The fleet-average CPU (`aggregate_cpu_utilization`)
stays between 9.11% and 9.91%. Last-value persistence (naive) beat every alternative I tried in a
rolling-origin backtest: MAE 0.206 percentage points (pp), about 2.2% of the mean level. Exponential
smoothing, damped trend, moving averages and drift were 15% to 58% worse in MAE.

The more useful operational finding is that the fleet mean hides the risk. Half of all readings are below
5% CPU, but 1.3% of readings exceed 80% and the top 1% of VMs account for 9.6% of all CPU. Capacity
decisions should therefore use a headroom margin on the fleet forecast plus tail and segment triggers
(Section 8), not the mean alone.

## 2. Data understanding and preprocessing

- **Source:** Atomity `data-v1` release, 10,000 VMs x 134 readings (1,340,000 rows). No other data is used.
- **Cadence and coverage:** every training step is exactly 300 s apart; each timestamp has exactly
  10,000 rows; 134 timestamps in total, so 110 for training and 24 for the holdout.
- **Quality:** 0 duplicates, 0 nulls and 0 out-of-range `avg_cpu` values. A reading is *valid* if
  `avg_cpu` is non-null and in [0, 100]; duplicate (timestamp, vm_id) rows are dropped. Because nothing was
  missing, no value was imputed and no resampling was done: modelling is at the native five-minute cadence,
  so no mapping back to the holdout timestamps is needed.
- **Targets:** `active_vm_count(t)` = distinct VM ids with a valid reading at t;
  `aggregate_cpu_utilization(t)` = mean of VM-level `avg_cpu` at t.
- **Holdout protocol:** the full target series is built first (as the assignment specifies), then split
  chronologically. The 24 holdout target values exist in memory at that point but are never read by
  fitting, selection or tuning, and the EDA and error analysis use only the first 110 timestamps.

## 3. Workload analysis

**Churn.** All 10,000 VMs are present at every training timestamp: 0 arrivals, 0 departures and 0 internal
gaps. The metadata agrees: the VMs were created at the start of the trace, and none is deleted before the
training cutoff. This is a property of how the release was sampled (VMs that lived through the whole
window), not evidence that real fleets are static. A consequence is that a fixed-cohort mean and the
full-fleet mean are identical, so every movement in the aggregate comes from VM behaviour, not from the
mix of VMs (composition effect = 0).

**Distribution** (`figures/cpu_distribution.png`). CPU use is heavily right-skewed:
mean 9.41%, median 5.12%, p90 19.7%, p99 87.4%, skew 3.7. 48.9% of readings are below 5%, 1.3% are above
80% (about 127 VMs at any moment), and the histogram has a secondary mode near 95% to 100%, a group of
near-saturated VMs. The top 1% of VMs (100 VMs) account for 9.6% of total CPU. *Why it matters:* the fleet
average understates the load that a latency-sensitive team has to protect.

**Temporal behaviour** (`figures/targets_train.png`). The fleet mean ranges over only 0.80 pp, with a
trend of -0.001 pp per step (about -0.12 pp over the window) and a step-to-step standard deviation of
0.157 pp. Its autocorrelation is 0.66 at lag 1, 0.33 at lag 12 (one hour) and -0.02 at lag 24 (two hours):
the next five minutes are well predicted by the last value, but little persists to the two-hour horizon.
The p90 across VMs is more volatile than the mean (about 19.5% rising to above 22% around steps 55 to 65,
then easing to about 18.5%).

**Segments** (`figures/segments_category.png`, `reports/tables/segment_*.csv`).

| Segment | Share of VMs | Mean CPU | p95 CPU |
|---|---|---|---|
| Delay-insensitive | 62% | 8.3% | 25.9% |
| Interactive | 31% | 11.5% | 49.2% |
| Unknown | 7% | 10.4% | 52.5% |
| 2 virtual cores | 75% | 9.8% | 39.2% |
| 4 virtual cores | 19% | 7.9% | 31.9% |
| 2 GB memory | 39% | 8.8% | 18.3% |
| 4 GB memory | 25% | 11.5% | 58.8% |

Interactive VMs run hotter and burstier than delay-insensitive ones (p95 49% vs 26%); 4 GB-memory VMs have
the heaviest tail (p95 59%). These segments deserve more headroom than the average.

**Peaks and lows.** The top-quartile steps of the fleet mean (above 9.51%) cluster at steps 53 to 72
(roughly 4.4 to 6 hours into the window); the bottom quartile (below 9.29%) is mostly in the second half.
The gap between peak and low periods is only 0.48 pp (about 5% relative), a small swing at fleet level.

## 4. Forecasting approach

Because the series has only 110 usable points, I used small statistical forecasters with no external
dependencies (NumPy and pandas only): naive (last value), moving averages over 6 and 12 steps, drift over
12 steps, simple exponential smoothing and damped Holt trend (parameters chosen by in-sample one-step SSE
on a small grid). They forecast each target directly at the five-minute cadence. No exogenous features are
used; every model sees only past target values available at the forecast origin. Predictions are clipped
to [0, 100] for CPU and to at least 0 for the VM count. I did not try gradient boosting or neural models:
with 110 points, no seasonality that can be learned from 11 hours, and lag structure that fades by two
hours, I judged they would mostly add overfitting risk.

## 5. Validation methodology

Rolling-origin evaluation inside the training window only, horizon 24 (the production problem: forecast
the next two hours from everything seen so far). Minimum training length 48, stride 12, giving origins at
steps 50, 62, 74 and 86, so 4 forecasts x 24 steps = 96 evaluation points. Windows overlap by half, so the
96 points are not independent. Model selection uses mean backtest MAE; ties go to the simpler model; the
chosen model is then refit on all 110 training points. Metrics: MAE, RMSE, peak-period MAE (steps where the
actual value is in the top quartile of the backtest actuals) and skill = 1 - MAE/MAE_naive. I report MAE
as a share of the mean level instead of MAPE; the series is never near zero, so MAPE would work, but the
level-relative MAE is simpler to read.

## 6. Results versus baseline

`aggregate_cpu_utilization` (backtest, 96 points):

| Model | MAE (pp) | RMSE (pp) | Peak MAE (pp) | Skill vs naive |
|---|---|---|---|---|
| **naive** | **0.206** | **0.254** | 0.259 | 0 |
| ses | 0.237 | 0.280 | 0.293 | -15.5% |
| holt_damped | 0.245 | 0.291 | 0.303 | -19.0% |
| mean_12 | 0.254 | 0.297 | 0.303 | -23.8% |
| mean_6 | 0.266 | 0.308 | 0.300 | -29.3% |
| drift_12 | 0.325 | 0.437 | 0.258 | -58.2% |

`active_vm_count`: every model has MAE 0 on all 96 backtest points because the series is constant, so
skill against naive is undefined (0/0). I forecast 10,000 and did not model it further.

Naive wins on MAE and RMSE; the only metric where another model edges it is drift's peak MAE (0.258 vs
0.259), a negligible difference offset by drift's much worse overall error. The honest reading is that
persistence is hard to beat on this smooth series, and smoothing helps only when there is noise to
average out. The differences between models are small in absolute terms (a few hundredths of a pp for the
smoothers), and with 4 overlapping origins they are suggestive, not conclusive.

## 7. Error and failure analysis

- **By horizon (naive):** MAE jumps from `0.042` pp at one step ahead to `0.248` at step 6,
  then is `0.223`, `0.328` and `0.198` pp at steps 12, 18 and 24. Error does not rise steadily:
  it moves from near zero at five minutes to about 0.2–0.3 pp beyond that. Bias is negative at
  steps 18 and 24 (-0.205 and -0.198 pp), meaning forecasts ran high on average. Per-step values
  are noisy because they come from only 4 overlapping windows (`reports/tables/error_by_horizon.csv`).
- **Peaks:** MAE in the top-quartile periods is 0.259 pp against `0.187` elsewhere
  (`error_peak_vs_nonpeak.csv`); persistence lags when the level moves.
- **By origin:** the largest errors occur for origins 50 (MAE 0.268 pp, bias +0.247 pp) and 62
  (MAE 0.297 pp, bias -0.283 pp). Since bias is actual minus forecast, origin 50 under-forecasts on
  average and origin 62 over-forecasts on average. The later origins have smaller errors (MAE 0.124
  and 0.132 pp) and negative bias (-0.113 and -0.117 pp).
- **Why trend and smoothing models did not help:** the autocorrelation drops to about zero by two hours,
  and the one visible regime shift (steps 53 to 72) is not preceded by a trend that damped Holt or drift
  can extrapolate. A 2-hour-ahead change of this kind cannot be anticipated from 9 hours of history.
- **Where it would fail:** a shift in the composition of the fleet (arrivals or departures), a daily
  cycle, or a spike in the heavy-tail VMs. None of these appears in this window, so I cannot validate
  behaviour under them.

## 8. Capacity-planning recommendation

**Policy:** provisioned capacity for each future step = forecast + margin, where the margin is a quantile
of the pooled backtest residuals (actual - forecast). Simulated over the backtest
(`reports/tables/headroom_policy.csv`):

| Quantile | Margin (pp) | Under-provision rate | Mean over-provision (pp) |
|---:|---:|---:|---:|
| 0.50 | -0.095 | 50.0% | 0.079 |
| 0.80 | 0.134 | 19.8% | 0.237 |
| 0.90 | 0.296 | 10.4% | 0.375 |
| 0.95 | 0.414 | 5.2% | 0.483 |
| 0.99 | 0.480 | 1.0% | 0.547 |

**Suggested setting:** use the 95th-percentile margin (`0.414` pp, about
`4.400%` of the mean level) for the fleet total. Because the residuals come from only
96 overlapping points, the 99th percentile is close to the single largest error; treat it as indicative.

**Trade-off:** under-provisioning risks throttled or slow services (SLA misses, incidents); over-provisioning
wastes paid capacity. Because the fleet mean is so stable, a small margin already covers most backtest
steps, so the cost of the margin is low in absolute terms. The asymmetry that matters is in the tail
below.

**Tail and segment rules (illustrative starting points, not validated here):**
1. Watch the share of VMs above 80% CPU. It averages 1.3% in training; investigate if it rises well above
   that (for example 1.5x), since that signals the tail growing even when the mean barely moves.
2. Give Interactive VMs and 4 GB-memory VMs extra headroom: their p95 CPU (49% and 59%) is about twice that
   of the delay-insensitive segment (26%).
3. Treat `active_vm_count` as constant only while real churn is absent; add a churn alert if the count
   changes between forecasts.

## 9. Limitations and future work

- **Short window:** 11.1 hours cannot reveal daily or weekly seasonality, so these results do not say how
  the workload behaves over a day.
- **Static fleet:** the release contains only VMs that lived through the entire window, so the constant
  VM count reflects the sampling, not the real workload. Churn modelling is impossible here.
- **Small evidence base:** 4 overlapping backtest origins and one 2-hour holdout; small model differences
  are not statistically established. No prediction intervals beyond residual quantiles were built.
- **Fleet mean only:** I did not forecast per segment or per VM, and did not test the tail triggers.
- **Future work:** segment-level forecasts (category and memory bucket), quantile or conformal intervals,
  anomaly detection around the steps 53 to 72 shift, and validation on a longer trace with churn and
  daily cycles.
