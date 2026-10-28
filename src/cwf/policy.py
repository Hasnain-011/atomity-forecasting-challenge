"""Headroom policy: provision = forecast + q-quantile of pooled backtest residuals."""
import numpy as np
import pandas as pd


def simulate(bt, quantiles=(0.5, 0.8, 0.9, 0.95, 0.99)):
    resid = bt["y"] - bt["yhat"]
    rows = []
    for q in quantiles:
        margin = float(resid.quantile(q))
        prov = bt["yhat"] + margin
        rows.append({"quantile": q, "margin": margin,
                     "under_rate": float((bt["y"] > prov).mean()),
                     "mean_over": float(np.maximum(prov - bt["y"], 0).mean()),
                     "mean_shortfall": float(np.maximum(bt["y"] - prov, 0).mean())})
    return pd.DataFrame(rows).set_index("quantile")


def margin_for(bt, q=0.95):
    return float((bt["y"] - bt["yhat"]).quantile(q))
