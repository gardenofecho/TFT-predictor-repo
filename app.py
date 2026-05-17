import streamlit as st
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- SERVER RESOURCE & THREAD PROTECTION ---
# Must happen before torch compiles operations to prevent multi-thread CPU spiking
import torch
torch.set_num_threads(1)

import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(page_title="TFT Live Asset Forecast", layout="wide")

st.title("📈 Live TFT Multi-Asset Forecast Engine")
st.markdown("This dashboard downloads real-time market data, generates technical indicators, and applies your trained Temporal Fusion Transformer (TFT) model to project future paths.")

# 2. Sidebar Interactive Handles
st.sidebar.header("🛠️ Macro Feature Scenario Handles")
st.sidebar.markdown("Adjust these sliders to simulate macro environments over the next 12 weeks.")

tuned_vix = st.sidebar.slider("Expected VIX Level (Fear Index)", min_value=10.0, max_value=60.0, value=18.0, step=0.5)
tuned_dxy_pct = st.sidebar.slider("Expected Weekly DXY Return (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
tuned_tnx_pct = st.sidebar.slider("Expected Weekly 10Y Yield Return (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.2)

# Convert percentage returns back to log returns expected by the model
tuned_dxy_return = np.log(1 + (tuned_dxy_pct / 100.0))
tuned_tnx_return = np.log(1 + (tuned_tnx_pct / 100.0))

# Baseline Configuration Parameters
LOOKBACK = 156
HORIZON = 12
TICKERS = ("SPY", "GLD")
WEEKLY_RULE = "W-FRI"
MACRO_TICKERS = {"vix_close": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX"}

# 3. Memory Exhaustion Protection (Cached Model Loader)
@st.cache_resource
def load_trained_tft(model_path="spy_gld_tft_model.ckpt"):
    try:
        return TemporalFusionTransformer.load_from_checkpoint(model_path)
    except Exception as e:
        return None

# Helpers for Technical Analysis Features
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_features(series: pd.Series) -> pd.DataFrame:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": macd - signal})

def fetch_yahoo_chart_frame(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int(datetime.now(timezone.utc).timestamp())
    params = urlencode({"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote_data = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    close = adjclose if adjclose is not None else quote_data.get("close")
    volume = quote_data.get("volume")
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
        "close": close,
        "volume": volume if volume is not None else np.nan,
    }).dropna(subset=["close"])
    return df.drop_duplicates("date").set_index("date").sort_index()

# 4. API Rate Limit Protection (Cached Data Frame Builder)
@st.cache_data(ttl=14400) # Re-downloads data once every 4 hours max
def get_live_processed_panel():
    asset_daily = {t: fetch_yahoo_chart_frame(t) for t in TICKERS}
    macro_daily = {name: fetch_yahoo_chart_frame(tk)[["close"]].rename(columns={"close": name}) for name, tk in MACRO_TICKERS.items()}
    
    weekly_closes, panel_rows = [], []
    for ticker, daily in asset_daily.items():
        daily = daily.copy()
        daily["daily_log_return"] = np.log(daily["close"] / daily["close"].shift(1))
        grouped = daily.resample(WEEKLY_RULE)
        weekly = pd.DataFrame({
            "price": grouped["close"].last(),
            "volume": grouped["volume"].sum(min_count=1),
            "trading_days": grouped["close"].count(),
            "realized_volatility": grouped["daily_log_return"].std(),
        }).dropna(subset=["price"])
        weekly["ticker"] = ticker
        weekly["weekly_log_return"] = np.log(weekly["price"] / weekly["price"].shift(1))
        weekly["volume_change"] = weekly["volume"].pct_change()
        weekly["rsi_14"] = rsi(weekly["price"], 14)
        weekly = weekly.join(macd_features(weekly["price"]))
        weekly["market_holidays"] = (weekly["trading_days"] < 5).astype(int)
        panel_rows.append(weekly.reset_index().rename(columns={weekly.reset_index().columns[0]: "date"}))
        weekly_closes.append(weekly[["price"]].rename(columns={"price": ticker}))
        
    price_wide = pd.concat(weekly_closes, axis=1).dropna()
    panel = pd.concat(panel_rows, ignore_index=True)
    
    macro = pd.DataFrame(index=price_wide.index)
    for name in MACRO_TICKERS.keys():
        macro[name] = macro_daily[name].resample(WEEKLY_RULE).last().reindex(price_wide.index).ffill()
    macro["dxy_return"] = np.log(macro["dxy"] / macro["dxy"].shift(1))
    macro["tnx_return"] = np.log(macro["tnx"] / macro["tnx"].shift(1))
    
    panel = panel.merge(macro.reset_index().rename(columns={macro.reset_index().columns[0]: "date"}), on="date", how="left")
    feature_cols = ["weekly_log_return", "volume_change", "realized_volatility", "rsi_14", "macd", "macd_signal", "macd_hist", "vix_close", "dxy_return", "tnx_return"]
    panel[feature_cols] = panel.groupby("ticker")[feature_cols].transform(lambda s: s.ffill().fillna(0.0))
    return price_wide, panel.sort_values(["ticker", "date"]).reset_index(drop=True)

def make_interactive_panel(observed_panel, price_wide, vix, dxy_ret, tnx_ret):
    panel = observed_panel.copy()
    panel["is_future"] = 0
    
    future_dates = pd.date_range(price_wide.index[-1] + pd.offsets.Week(weekday=4), periods=HORIZON, freq=WEEKLY_RULE)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=future_dates.min() - pd.Timedelta(days=6), end=future_dates.max())
    
    future_rows = []
    for ticker in TICKERS:
        last = panel[panel["ticker"] == ticker].sort_values("date").iloc[-1]
        for date in future_dates:
            week_start = date - pd.Timedelta(days=6)
            is_holiday = int(((holidays >= week_start) & (holidays <= date)).any())
            future_rows.append({
                "date": date, "ticker": ticker, "price": last["price"], "volume": last["volume"],
                "trading_days": 5 - is_holiday, "realized_volatility": last["realized_volatility"],
                "weekly_log_return": 0.0, "volume_change": 0.0, "rsi_14": last["rsi_14"],
                "macd": last["macd"], "macd_signal": last["macd_signal"], "macd_hist": last["macd_hist"],
                "market_holidays": is_holiday,
                "vix_close": vix, "dxy_return": dxy_ret, "tnx_return": tnx_ret, "is_future": 1
            })
            
    panel = pd.concat([panel, pd.DataFrame(future_rows)], ignore_index=True)
    base_date = panel["date"].min()
    panel["time_idx"] = ((panel["date"] - base_date).dt.days // 7).astype(int)
    panel["weekofyear"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["sin_week"] = np.sin(2 * np.pi * panel["weekofyear"] / 52.0)
    panel["cos_week"] = np.cos(2 * np.pi * panel["weekofyear"] / 52.0)
    return panel.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
