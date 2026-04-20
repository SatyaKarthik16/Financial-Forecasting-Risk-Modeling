"""
Revenue forecasting using ARIMA with holdout evaluation and confidence intervals.
"""

import argparse
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)


def load_monthly(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath, parse_dates=["Date"])
    s = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    s.index = s.index.to_timestamp()
    s.name = "Revenue"
    return s


def select_order(series: pd.Series, d: int):
    best_aic = np.inf
    best = (1, d, 1)
    for p, q in itertools.product(range(4), range(4)):
        try:
            res = ARIMA(series, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best = (p, d, q)
        except Exception:
            continue
    return best, best_aic


def main(filepath: str, horizon: int):
    series = load_monthly(filepath)
    print(f"[Forecasting] Monthly series: {len(series)} periods")

    pval = adfuller(series.dropna())[1]
    d = 0 if pval < 0.05 else 1
    print(f"[Forecasting] ADF p-value={pval:.4f}, using d={d}")

    order, aic = select_order(series, d)
    print(f"[Forecasting] Best ARIMA{order} with AIC={aic:.2f}")

    holdout = min(6, max(3, len(series) // 4))
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    hmodel = ARIMA(train, order=order).fit()
    hpred = hmodel.forecast(steps=holdout)
    hpred.index = test.index

    mae = np.mean(np.abs(test.values - hpred.values))
    mape = np.mean(np.abs((test.values - hpred.values) / test.values)) * 100
    print(f"[Forecasting] Holdout MAE=${mae:,.2f}, MAPE={mape:.2f}%")

    model = ARIMA(series, order=order).fit()
    out = model.get_forecast(steps=horizon)
    mean = out.predicted_mean
    ci = out.conf_int(alpha=0.05)

    future_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    mean.index = future_index
    ci.index = future_index

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, label="Historical", color="#1f77b4")
    ax.plot(mean.index, mean.values, label="Forecast", color="#2ca02c")
    ax.fill_between(mean.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, color="#2ca02c", label="95% CI")
    ax.legend()
    ax.set_title(f"Revenue Forecast ARIMA{order}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("images/revenue_forecast_arima.png", dpi=150)
    plt.close()

    tbl = pd.DataFrame({
        "Month": mean.index.strftime("%b %Y"),
        "Forecast": mean.round(2).values,
        "Lower95": ci.iloc[:, 0].round(2).values,
        "Upper95": ci.iloc[:, 1].round(2).values,
    })
    tbl.to_csv("revenue_forecast.csv", index=False)
    print("Saved revenue_forecast.csv and images/revenue_forecast_arima.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_transactions.csv")
    p.add_argument("--horizon", type=int, default=6)
    a = p.parse_args()
    main(a.data, a.horizon)
