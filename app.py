import streamlit as st
import matplotlib.pyplot as plt

# ... [Previous data processing and model inference code] ...

# --- STREAMLIT DISPLAY MATCHING KAGGLE NOTEBOOK ---
st.subheader("🔮 SPY and GLD Forecast Paths")

# Match your notebook's subplot setup: fig, axes = plt.subplots(len(DFB.tickers), 1, figsize=(12, 7), sharex=True)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

tickers = ['SPY', 'GLD']

for i, ticker in enumerate(tickers):
    ax = axes[i]
    
    # 1. Plot historical context (Actual)
    # Replace history_df with your actual history slice variable
    ax.plot(history_df.index, history_df[ticker], label='Actual', color='gray', alpha=0.7)
    
    # 2. Plot model predictions (TFT Forecast)
    # Replace forecast_df with your model's processed outputs
    ax.plot(forecast_df.index, forecast_df[ticker], label='TFT Forecast', color='dodgerblue', linestyle='--')
    
    # Visual formatting matching Kaggle stylings
    ax.set_title(f"{ticker} Multi-step Forecast")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# Render the matplotlib figure directly inside Streamlit
st.pyplot(fig)
