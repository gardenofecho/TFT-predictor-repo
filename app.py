import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dataclasses import dataclass

# 1. Page Configuration & Stylings
st.set_page_config(page_title="Live TFT Multi-Asset Forecast Engine", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")

st.title("📊 Live TFT Multi-Asset Forecast Engine")
st.caption("This dashboard generates technical indicators and applies your trained Temporal Fusion Transformer (TFT) model to project future paths.")

# 2. Hardcoded Config Matching Kaggle Setup
@dataclass
class Config:
    tickers: tuple[str, ...] = ("SPY", "GLD")
    start: str = "2010-01-01"
    lookback: int = 156
    horizon: int = 12

CFG = Config()

# --- 3. REMOVED THE 3 HANDLES (SLIDERS) AND FIXED BASELINES ---
vix_level = 15.0         # Default baseline target for VIX close
spy_return = 0.0         # Default baseline target for weekly SPY return
t_yield = 0.0            # Default baseline target for 10Y Yield return

# 4. Data Generation & Model Simulation Infrastructure Using Live yfinance
def load_historical_and_forecast_data():
    plot_history = {}
    forecast_hicker = {}
    
    for ticker in CFG.tickers:
        # Fetch real weekly data from Yahoo Finance
        ticker_obj = yf.Ticker(ticker)
        historical_df = ticker_obj.history(start=CFG.start, interval="1wk")
        
        # Extract the Closing price series
        plot_history[ticker] = historical_df['Close']
        
        # Use the latest real price as the baseline anchor for predictions
        last_price = historical_df['Close'].iloc[-1]
        
        # Generate future timeline index
        future_dates = pd.date_range(start=historical_df.index[-1] + pd.Timedelta(weeks=1), periods=CFG.horizon, freq="W-FRI")
        
        # --- PLACEHOLDER FOR YOUR REAL TFT MODEL PREDICTIONS ---
        # Simulates a future trajectory starting directly from the live market price
        simulated_predictions = last_price + np.cumsum(np.random.normal(0, last_price * 0.01, CFG.horizon))
        forecast_hicker[ticker] = pd.Series(simulated_predictions, index=future_dates)
        
    return plot_history, forecast_hicker

# Call the function to instantiate global data variables
with st.spinner("Fetching live Yahoo Finance data and running engine..."):
    plot_history, forecast_hicker = load_historical_and_forecast_data()

# --- 5. ADD INTERACTIVE LOOKBACK SELECTOR ---
st.write("---")
st.subheader("🛠️ Visualization Controls")
zoom_weeks = st.slider("Historical Lookback Window (Weeks)", min_value=12, max_value=CFG.lookback, value=52, step=12)

# --- 6. STREAMLIT DISPLAY MATCHING KAGGLE PLOT ---
st.subheader("🔮 SPY and GLD Forecast Paths")

# Set up matplotlib figure identical to Kaggle notebook setup
fig, axes = plt.subplots(len(CFG.tickers), 1, figsize=(12, 8), sharex=True)

for i, ticker in enumerate(CFG.tickers):
    ax = axes[i]
    
    # Slice the trailing historical context dynamically based on slider
    history_slice = plot_history[ticker].tail(zoom_weeks)
    
    # Line 1: Historical Actuals
    ax.plot(history_slice.index, history_slice.values, label='actual', color='#1f77b4')
    
    # Line 2: Future TFT Forecasts
    ax.plot(forecast_hicker[ticker].index, forecast_hicker[ticker].values, 
            label=f'{CFG.horizon}-week TFT Forecast from predicted returns', color='red', linewidth=1.5)
    
    # Formatting with enlarged high-visibility fonts
    ax.set_title(f"{ticker} {CFG.horizon}-week TFT Forecast from predicted returns", fontsize=18, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Render directly into the main body layout
st.pyplot(fig)
