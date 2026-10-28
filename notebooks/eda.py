# %% [markdown]
# # EDA: Cloud Workload Forecasting (training window only)
# Holdout discipline: the timestamp list is taken from raw rows and cut to the first n-24.
# No holdout target value is computed or plotted anywhere in this notebook.

# %%
import json, os
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython()  # running inside a notebook: keep the inline backend
except NameError:
    matplotlib.use("Agg")  # plain script: save figures, never wait on a window
import matplotlib.pyplot as plt
from cwf import data as D

DATA_DIR = os.environ.get("CWF_DATA", "challenge_data")
FIG, TAB = "reports/figures", "reports/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
summary = {}

def savefig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=130)
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()

# %% [markdown]
# ## 1. Load, cadence, data quality

# %%
raw = D.load_readings(DATA_DIR)
meta = pd.read_csv(f"{DATA_DIR}/vm_metadata.csv.gz")
all_ts = np.sort(raw["timestamp"].unique())
train_ts = all_ts[:-D.HOLDOUT]
raw_tr = raw[raw["timestamp"].isin(train_ts)]          # training rows only from here on
df, rep = D.clean(raw_tr)

steps = pd.Series(train_ts).diff().dropna()
rows_per_ts = raw_tr.groupby("timestamp").size()
summary["cadence"] = {"n_train_timestamps": len(train_ts),
                      "n_total_timestamps": len(all_ts),
                      "step_counts_seconds": {int(k): int(v) for k, v in steps.value_counts().items()},
                      "rows_per_ts_min_max": [int(rows_per_ts.min()), int(rows_per_ts.max())]}
summary["quality"] = {**rep, "null_avg_cpu": int(raw_tr["avg_cpu"].isna().sum()),
                      "vms_in_train": int(raw_tr["vm_id"].nunique())}
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 2. VM lifetimes and churn (inferred from readings; only information observable before the cutoff)

# %%
idx = pd.Series(range(len(train_ts)), index=train_ts)
df = df.assign(i=df["timestamp"].map(idx))
life = df.groupby("vm_id")["i"].agg(first="min", last="max", n="count")
life["span"] = life["last"] - life["first"] + 1
life["gaps"] = life["span"] - life["n"]
full_cohort = life.index[(life["first"] == 0) & (life["last"] == len(train_ts) - 1) & (life["gaps"] == 0)]
arrivals = life.groupby("first").size().reindex(range(len(train_ts)), fill_value=0)
departures = life.groupby("last").size().reindex(range(len(train_ts)), fill_value=0)
summary["churn"] = {"vms_present_at_every_ts": int(len(full_cohort)),
                    "share_full_cohort": float(len(full_cohort) / len(life)),
                    "arrivals_after_first_ts": int(arrivals.iloc[1:].sum()),
                    "departures_before_last_ts": int(departures.iloc[:-1].sum()),
                    "vms_with_internal_gaps": int((life["gaps"] > 0).sum())}
# metadata: only use what is knowable at the cutoff (units assumed to match reading timestamps; verify)
cutoff = train_ts[-1]
m = meta.set_index("vm_id")
summary["metadata_at_cutoff"] = {
    "created_before_first_ts": float((m["timestamp_vm_created"] < train_ts[0]).mean()),
    "created_within_train": float(m["timestamp_vm_created"].between(train_ts[0], cutoff).mean()),
    "deleted_by_cutoff": float((m["timestamp_vm_deleted"] <= cutoff).mean())}
print(json.dumps({k: summary[k] for k in ("churn", "metadata_at_cutoff")}, indent=2))
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(len(train_ts)), arrivals.values, label="first seen")
ax.bar(range(len(train_ts)), -departures.values, label="last seen")
ax.set_xlabel("train step"); ax.set_ylabel("VMs"); ax.legend(); ax.set_title("Arrivals / departures")
savefig("churn")

# %% [markdown]
# ## 3. Targets on the training window, plus level and variability

# %%
g = df.groupby("timestamp")["avg_cpu"]
ts_df = pd.DataFrame({"active": df.groupby("timestamp")["vm_id"].nunique(),
                      "mean": g.mean(), "median": g.median(),
                      "p90": g.quantile(.9), "p95": g.quantile(.95)})
ts_df.to_csv(f"{TAB}/train_series.csv")
pooled = df["avg_cpu"]
summary["cpu_distribution"] = {"mean": float(pooled.mean()), "median": float(pooled.median()),
    "p90": float(pooled.quantile(.9)), "p99": float(pooled.quantile(.99)),
    "share_below_5pct": float((pooled < 5).mean()), "share_above_80pct": float((pooled > 80).mean()),
    "skew": float(pooled.skew())}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].hist(pooled, bins=50); axs[0].set_title("Pooled avg_cpu"); axs[0].set_yscale("log")
for c in ["mean", "median", "p90"]:
    axs[1].plot(range(len(ts_df)), ts_df[c], label=c)
axs[1].legend(); axs[1].set_title("Per-timestamp CPU statistics"); axs[1].set_xlabel("train step")
savefig("cpu_distribution")

y = ts_df["mean"].to_numpy(); n = ts_df["active"].to_numpy()
def acf(x, k):
    x = x - x.mean(); return float((x[k:] * x[:-k]).sum() / (x * x).sum())
summary["temporal"] = {
    "mean_cpu_range": [float(y.min()), float(y.max())],
    "mean_cpu_slope_per_step": float(np.polyfit(range(len(y)), y, 1)[0]),
    "mean_cpu_diff_std": float(np.diff(y).std()),
    "mean_cpu_acf_lag1_12_24": [acf(y, k) for k in (1, 12, 24)],
    "active_range": [float(n.min()), float(n.max())],
    "active_diff_std": float(np.diff(n).std())}
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(y); ax[0].set_ylabel("mean avg_cpu (%)")
ax[1].plot(n); ax[1].set_ylabel("active VMs"); ax[1].set_xlabel("train step")
savefig("targets_train")
print(json.dumps({k: summary[k] for k in ("cpu_distribution", "temporal")}, indent=2))

# %% [markdown]
# ## 4. Composition vs behavior: fixed cohort vs full active set

# %%
cohort = df[df["vm_id"].isin(full_cohort)].groupby("timestamp")["avg_cpu"].mean()
cmp = pd.DataFrame({"full": ts_df["mean"], "cohort": cohort})
cmp["composition_effect"] = cmp["full"] - cmp["cohort"]
summary["composition"] = {"max_abs_composition_effect": float(cmp["composition_effect"].abs().max()),
                          "corr_full_vs_cohort": float(cmp[["full", "cohort"]].corr().iloc[0, 1])}
fig, ax = plt.subplots(figsize=(8, 3))
cmp[["full", "cohort"]].reset_index(drop=True).plot(ax=ax); ax.set_ylabel("mean avg_cpu (%)")
ax.set_title("Full active set vs fixed cohort"); savefig("cohort_vs_full")
print(summary["composition"])

# %% [markdown]
# ## 5. Segments (metadata attributes are static per VM, so no leakage)

# %%
seg = df.merge(meta[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]], on="vm_id", how="left")
summary["metadata_unmatched_rows"] = int(seg["vm_category"].isna().sum())
seg_tables = {}
for col in ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]:
    t = seg.groupby(col)["avg_cpu"].agg(["count", "mean", "median"])
    t["n_vms"] = seg.groupby(col)["vm_id"].nunique()
    t["share_of_vms"] = t["n_vms"] / seg["vm_id"].nunique()
    t["p95"] = seg.groupby(col)["avg_cpu"].quantile(.95)
    seg_tables[col] = t; t.to_csv(f"{TAB}/segment_{col}.csv"); print(t.round(2), "\n")
fig, ax = plt.subplots(figsize=(8, 3.2))
for k, grp in seg.groupby("vm_category"):
    grp.groupby("timestamp")["avg_cpu"].mean().reset_index(drop=True).plot(ax=ax, label=str(k))
ax.legend(); ax.set_title("Mean CPU by vm_category"); ax.set_xlabel("train step"); savefig("segments_category")

# %% [markdown]
# ## 6. Peaks, lows, and concentration

# %%
q_hi, q_lo = np.quantile(y, .75), np.quantile(y, .25)
peak_steps = np.where(y >= q_hi)[0]; low_steps = np.where(y <= q_lo)[0]
vm_mean = df.groupby("vm_id")["avg_cpu"].mean().sort_values(ascending=False)
top1 = vm_mean.index[: max(1, int(len(vm_mean) * 0.01))]
total = df["avg_cpu"].sum()
summary["peaks"] = {"peak_threshold": float(q_hi), "low_threshold": float(q_lo),
    "peak_steps": peak_steps.tolist()[:40], "low_steps": low_steps.tolist()[:40],
    "top1pct_vm_share_of_cpu_mass": float(df[df["vm_id"].isin(top1)]["avg_cpu"].sum() / total),
    "peak_vs_low_mean_gap": float(y[peak_steps].mean() - y[low_steps].mean())}
print(json.dumps(summary["peaks"], indent=2))

# %%
with open(f"{TAB}/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("saved", f"{TAB}/eda_summary.json")
