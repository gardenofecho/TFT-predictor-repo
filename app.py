# 5. Data Generation & Model Inference Pipeline
def load_historical_and_forecast_data():
    plot_history = {}
    forecast_hicker = {}
    global tft_model
    
    for ticker in CFG.tickers:
        try:
            # Enforce flat, single-level columns directly inside the query download parameter
            historical_df = yf.download(
                ticker, 
                start=CFG.start, 
                interval="1d", 
                multi_level_index=False,  # Forces clean 1D column keys directly from the API
                progress=False
            )
        except Exception as api_err:
            historical_df = pd.DataFrame()
            
        # Resample daily data to weekly Friday blocks to match training parameters
        if not historical_df.empty and len(historical_df) > 10:
            historical_df = historical_df.resample('W-FRI').last()
            historical_df = historical_df.dropna(subset=['Close'])
        
        # Fallback safeguard if DataFrame initialization fails or returns empty rows
        if historical_df.empty or len(historical_df) < 10:
            st.sidebar.error(f"Failed to parse live data for {ticker}. Using simulation mode.")
            date_range = pd.date_range(end=pd.Timestamp.now(), periods=200, freq="W-FRI")
            base_val = 450 if ticker == "SPY" else 200
            sim_prices = base_val + np.cumsum(np.random.normal(0.2, 2.5, len(date_range)))
            historical_df = pd.DataFrame({'Close': sim_prices}, index=date_range)
        
        # Strip timezone metadata safely to avoid plotting index mismatches
        historical_df.index = pd.to_datetime(historical_df.index).tz_localize(None)
        plot_history[ticker] = historical_df['Close']
        
        # Safe extraction now that index dimensions are fully isolated
        last_price = float(historical_df['Close'].values[-1])
        last_date = historical_df.index[-1]
        
        # Build forward timeline projection windows
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=CFG.horizon, freq="W-FRI")
        simulated_predictions = []
        current_price = last_price
        
        if tft_model is not None:
            try:
                # Calculate logarithmic returns to match neural network targets
                history_window = historical_df.tail(CFG.lookback).copy()
                history_window['weekly_log_return'] = np.log(history_window['Close'] / history_window['Close'].shift(1))
                history_window['weekly_log_return'] = history_window['weekly_log_return'].fillna(0)
                
                # Reshape raw features into active PyTorch tensor structures
                input_array = history_window['weekly_log_return'].values[-CFG.lookback:]
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                
                with torch.no_grad():
                    raw_prediction = tft_model(input_tensor)
                    predicted_returns = raw_prediction.output[0, :, 0].cpu().numpy()
                
                # Loop through outputs and reconstruct relative pricing trajectory
                for log_ret in predicted_returns[:CFG.horizon]:
                    current_price = current_price * np.exp(log_ret)
                    simulated_predictions.append(current_price)
            except Exception as inference_error:
                pass
                
        # Fill with a standard random walk scenario if model inference fails
        if len(simulated_predictions) == 0:
            simulated_predictions = []
            current_price = last_price
            for _ in range(CFG.horizon):
                current_price = current_price * (1 + np.random.normal(0.001, 0.015))
                simulated_predictions.append(current_price)
                    
        forecast_hicker[ticker] = pd.Series(simulated_predictions, index=future_dates)
        
    return plot_history, forecast_hicker
