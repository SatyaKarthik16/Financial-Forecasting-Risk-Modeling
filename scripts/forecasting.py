import argparse, os, warnings, itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")
os.makedirs("images", exist_ok=True)

def load_monthly_revenue(filepath):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    monthly.index = monthly.index.to_timestamp()
    monthly.name = "Revenue"
    return monthly

def main(filepath, horizon=6):
    series = load_monthly_revenue(filepath)
    print(f"[Forecasting] {len(series)} months: {series.index[0].strftime('%b %Y')} to {series.index[-1].strftime('%b %Y')}")
    p_val = adfuller(series)[1]
    d = 0 if p_val < 0.05 else 1
    print(f"[Forecasting] ADF p={p_val:.4f} -> d={d}")
    best_aic, best_order = np.inf, (1, d, 1)
    for p, q in itertools.product(range(4), range(4)):
        try:
            res = ARIMA(series, order=(p,d,q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p,d,q)
        except: continue
    print(f"[Forecasting] Best ARIMA{best_order} AIC={best_aic:.2f}")
    holdout = min(6, len(series)//4)
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    ho_fc = ARIMA(train, order=best_order).fit().forecast(steps=holdout)
    ho_fc.index = test.index
    mae  = np.mean(np.abs(test.values - ho_fc.values))
    mape = np.mean(np.abs((test.values - ho_fc.values)/test.values))*100
    print(f"[Forecasting] Hold-out MAE=${mae:,.2f}  MAPE={mape:.2f}%")
    fitted = ARIMA(series, order=best_order).fit()
    fc_result = fitted.get_forecast(steps=horizon)
    fc_mean = fc_result.predicted_mean
    conf_int = fc_result.conf_int(alpha=0.05)
    future_idx = pd.date_range(start=series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc_mean.index = conf_int.index = future_idx
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(series.index, series.values/1e6, color="#1f77b4", linewidth=2, marker="o", markersize=3, label="Historical")
    ax.plot(fc_mean.index, fc_mean.values/1e6, color="#2ca02c", linewidth=2.5, marker="D", markersize=6, label="Forecast")
    ax.fill_between(fc_mean.index, conf_int.iloc[:,0]/1e6, conf_int.iloc[:,1]/1e6, alpha=0.2, color="#2ca02c", label="95% CI")
    ax.axvline(fc_mean.index[0], color="grey", linestyle="--", alpha=0.6)
    ax.set_title(f"Monthly Revenue — ARIMA{best_order} Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue ($M)"); ax.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig("images/revenue_forecast_arima.png", dpi=150)
    plt.show()
    tbl = pd.DataFrame({"Month":fc_mean.index.strftime("%b %Y"),"Forecast($M)":(fc_mean/1e6).round(3).values,
        "Lower CI($M)":(conf_int.iloc[:,0]/1e6).round(3).values,"Upper CI($M)":(conf_int.iloc[:,1]/1e6).round(3).values})
    tbl.to_csv("revenue_forecast.csv", index=False)
    print("[Forecasting] Saved revenue_forecast.csv and images/revenue_forecast_arima.png")
    print(tbl.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_transactions.csv")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()
    main(args.data, args.horizon)
