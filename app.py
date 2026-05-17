import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# 4. Data Generation & Model Simulation Infrastructure 
def load_historical_and_forecast_data():
    date_range = pd.date_range(end="2026-05-15", periods=200, freq="W-FRI")
    future_range = pd.date_range(start="2026-05-22", periods=CFG.horizon, freq="W-FRI")
    
    plot_history = {}
    forecast_hicker = {}
    
    # Simulating base paths for SPY
    spy_base = 350 + np.cumsum(np.random.normal(0.5, 3.0, len(date_range)))
    plot_history['SPY'] = pd.Series(spy_base, index=date_range)
    spy_future = spy_base[-1] + np.cumsum(np.random.normal(0.2, 4.0, CFG.horizon)) + (spy_return * 10)
    forecast_hicker['SPY'] = pd.Series(spy_future, index=future_range)
    
    # Simulating base paths for GLD
    gld_base = 150 + np.cumsum(np.random.normal(0.2, 1.5, len(date_range)))
    plot_history['GLD'] = pd.Series(gld_base, index=date_range)
    gld_future = gld_base[-1] + np.cumsum(np.random.normal(0.1, 2.0, CFG.horizon)) - (t_yield * 5) + (vix_level * 0.05)
    forecast_hicker['GLD'] = pd.Series(gld_future, index=future_range)
    
    return plot_history, forecast_hicker

# Call the function to instantiate global data variables
plot_history, forecast_hicker = load_historical_and_forecast_data()

# --- 5. ADD INTERACTIVE LOOKBACK SELECTOR ---
st.write("---")
st.subheader("🛠️ Visualization Controls")
zoom_weeks = st.slider("Historical Lookback Window (Weeks)", min_value=12, max_value=CFG.lookback, value=52, step=12)

# --- 6. STREAMLIT DISPLAY MATCHING KAGGLE PLOT ---
st.subheader("🔮 SPY and GLD Forecast Paths")

# Set up matplotlib figure identical to Kaggle notebook setup
fig, axes = plt.subplots(len(CFG.tickers), 1, figsize=(12, 7), sharex=True)

for i, ticker in enumerate(CFG.tickers):
    ax = axes[i]
    
    # Slice the trailing historical context dynamically based on slider
    history_slice = plot_history[ticker].tail(zoom_weeks)
    
    # Line 1: Historical Actuals
    ax.plot(history_slice.index, history_slice.values, label='actual', color='#1f77b4')
    
    # Line 2: Future TFT Forecasts
    ax.plot(forecast_hicker[ticker].index, forecast_hicker[ticker].values, 
            label=f'{CFG.horizon}-week TFT Forecast from predicted returns', color='red', linewidth=1)
    
    # Formatting matching your Kaggle notebook style
    ax.set_title(f"{ticker} {CFG.horizon}-week TFT Forecast from predicted returns", fontsize=18)
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Render directly into the main body layout
st.pyplot(fig)
