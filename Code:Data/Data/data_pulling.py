import yfinance as yf
import pandas as pd
from pandas_datareader import data as web
import numpy as np

# define ETFs and get monthly close prices
tickers = ["SPY", "QQQ", "IWM", "IWD", "IWF"]
prices = pd.DataFrame()

for ticker in tickers:
    df = yf.Ticker(ticker).history(period="max", interval="1mo")[["Close"]]
    df.rename(columns={"Close": ticker}, inplace=True)
    prices = pd.concat([prices, df], axis=1)

prices.index = prices.index.tz_localize(None) # remove timezone info from pricing index to match factor index

prices.to_csv("ETF_monthly_prices.csv")

returns = prices.pct_change()
returns.to_csv("ETF_monthly_returns.csv")

# get fama–french 5 factors and risk-free rate
ff = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=prices.index.min().strftime("%Y-%m"))[0]

ff.index = ff.index.to_timestamp()
ff = ff.div(100)


ff.to_csv("FamaFrench_5Factors.csv")

# Merge returns and factors into a single panel
panel = returns.merge(ff, left_index=True, right_index=True, how="inner")

# Fetch VIX from FRED and flag low/neutral/high via 6-month MA
vix = web.DataReader("VIXCLS", "fred", start=panel.index.min(), end=panel.index.max())
vix_m = vix.resample("MS").first()
vix_m = vix_m.reindex(panel.index, method="ffill")

# Compute 6-month moving average of VIX and percent deviation
vix_ma6 = vix_m["VIXCLS"].rolling(window=6, min_periods=1).mean()
pct_vix = (vix_m["VIXCLS"] - vix_ma6) / vix_ma6

# Classify as Low, Neutral, or High
panel["VIX"] = np.where(
    pct_vix < -0.10, "Low",
    np.where(pct_vix >  0.10, "High", "Neutral")
)


# Define economic Bull/Bear/Neutral based on 20% drawdown and 20% recovery
price = prices["SPY"].reindex(panel.index, method="ffill")
running_max = price.cummax()
drawdown   = price / running_max - 1.0

labels = []
state = "Neutral"
current_trough = None
for date in panel.index:
    p = price.loc[date]
    dd = drawdown.loc[date]
    if state != "Bear":
        # From Neutral or Bull, check for entry into Bear
        if dd <= -0.10:
            state = "Bear"
            current_trough = p
        # From Bull, allow reversion to Neutral when drawdown recovers above -10%
        elif state == "Bull" and dd > -0.10:
            state = "Neutral"
    else:
        # In Bear, update trough and check for 10% recovery to Bull
        if p < current_trough:
            current_trough = p
        if (p / current_trough - 1.0) >= 0.10:
            state = "Bull"
    labels.append(state)

panel["Bull_Bear"] = labels

# Export only factor columns, RF, VIX_Regime, and Bull_Bear to CSV
regimes = panel[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "VIX", "Bull_Bear"]]
regimes.to_csv("regimes.csv")
