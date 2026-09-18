import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Try to import Prophet, handle fallback if not installed
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    # Suppress prophet logs
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    PROPHET_AVAILABLE = True
    logger.info("Prophet is successfully imported and available.")
except ImportError:
    logger.warning("Prophet not found. Forecasting engine will fall back to Holt-Winters Exponential Smoothing as alternative.")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def prepare_time_series(df, period_type='Daily'):
    """Aggregate transaction data into a continuous time series."""
    df_ts = df.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['InvoiceDate']).dt.date
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    
    # Set Date as Index
    df_ts.set_index('Date', inplace=True)
    
    # Resample based on period type
    if period_type == 'Daily':
        ts = df_ts['Revenue'].resample('D').sum().fillna(0)
    elif period_type == 'Weekly':
        ts = df_ts['Revenue'].resample('W').sum().fillna(0)
    elif period_type == 'Monthly':
        ts = df_ts['Revenue'].resample('MS').sum().fillna(0)
    else:
        raise ValueError("Invalid period_type. Must be Daily, Weekly, or Monthly")
        
    return ts

def run_arima_forecast(ts, train_len, forecast_periods):
    """Fit ARIMA and return test predictions, evaluation metrics, and future forecasts."""
    # Resample index should have frequency
    ts = ts.asfreq(ts.index.inferred_freq or ('D' if ts.index.name=='Daily' else 'W'))
    
    # Train/Test Split
    train = ts.iloc[:train_len]
    test = ts.iloc[train_len:]
    
    # Grid search a small set of order options for the best AIC
    best_aic = float('inf')
    best_order = (1, 1, 1)
    orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1)]
    
    for order in orders:
        try:
            model = ARIMA(train, order=order)
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
        except Exception:
            continue
            
    # Train evaluation
    logger.info("ARIMA chosen best order: %s", str(best_order))
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()
    
    # Predict on test
    test_pred = model_fit.forecast(steps=len(test))
    test_pred = np.clip(test_pred, 0, None) # Revenue shouldn't be negative
    
    # Compute metrics
    mae = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))
    mape = calculate_mape(test, test_pred)
    
    # Fit on ALL data for future forecasting
    full_model = ARIMA(ts, order=best_order)
    full_fit = full_model.fit()
    
    # Forecast future
    forecast_res = full_fit.get_forecast(steps=forecast_periods)
    forecast_mean = np.clip(forecast_res.predicted_mean, 0, None)
    
    # Confidence Intervals (95% default)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_ci = np.clip(conf_int.iloc[:, 0], 0, None)
    upper_ci = conf_int.iloc[:, 1]
    
    future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1) if ts.index.inferred_freq == 'D' else ts.index[-1] + pd.Timedelta(weeks=1), 
                                 periods=forecast_periods, 
                                 freq=ts.index.inferred_freq)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_mean.values,
        'Lower_CI': lower_ci.values,
        'Upper_CI': upper_ci.values
    })
    
    metrics = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}
    return forecast_df, metrics, test_pred

def run_prophet_forecast(ts, train_len, forecast_periods):
    """Fit Prophet (or Holt-Winters fallback) and return predictions and metrics."""
    if not PROPHET_AVAILABLE:
        # Fallback to Holt-Winters Exponential Smoothing
        return run_holt_winters_forecast(ts, train_len, forecast_periods)
        
    # Prepare Prophet format
    df_prophet = ts.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    train = df_prophet.iloc[:train_len]
    test = df_prophet.iloc[train_len:]
    
    # Fit Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train)
    
    # Predict on test
    future_test = model.make_future_dataframe(periods=len(test), freq=ts.index.inferred_freq or 'D')
    forecast_test = model.predict(future_test)
    test_pred = forecast_test['yhat'].iloc[train_len:].values
    test_pred = np.clip(test_pred, 0, None)
    
    # Metrics
    y_test = test['y'].values
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mape = calculate_mape(y_test, test_pred)
    
    # Fit full model
    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    full_model.fit(df_prophet)
    
    # Forecast future
    future_dates = full_model.make_future_dataframe(periods=forecast_periods, freq=ts.index.inferred_freq or 'D')
    forecast_full = full_model.predict(future_dates)
    
    future_forecast = forecast_full.iloc[-forecast_periods:]
    
    forecast_df = pd.DataFrame({
        'Date': future_forecast['ds'].values,
        'Forecast': np.clip(future_forecast['yhat'].values, 0, None),
        'Lower_CI': np.clip(future_forecast['yhat_lower'].values, 0, None),
        'Upper_CI': future_forecast['yhat_upper'].values
    })
    
    metrics = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}
    return forecast_df, metrics, test_pred

def run_holt_winters_forecast(ts, train_len, forecast_periods):
    """Fit Holt-Winters Exponential Smoothing and return predictions and metrics."""
    train = ts.iloc[:train_len]
    test = ts.iloc[train_len:]
    
    # Fit Holt-Winters
    try:
        model = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add')
        model_fit = model.fit()
    except Exception:
        # Simplistic fallback if seasonal decomposition fails
        model = ExponentialSmoothing(train, trend='add')
        model_fit = model.fit()
        
    test_pred = model_fit.forecast(steps=len(test))
    test_pred = np.clip(test_pred, 0, None)
    
    mae = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))
    mape = calculate_mape(test, test_pred)
    
    # Fit on all data
    try:
        full_model = ExponentialSmoothing(ts, seasonal_periods=7, trend='add', seasonal='add')
        full_fit = full_model.fit()
    except Exception:
        full_model = ExponentialSmoothing(ts, trend='add')
        full_fit = full_model.fit()
        
    forecast_mean = np.clip(full_fit.forecast(steps=forecast_periods), 0, None)
    
    # Construct synthetic confidence intervals since statsmodels HW prediction intervals can be unstable
    # We use residual standard error
    residuals = full_fit.resid
    std_error = np.std(residuals)
    
    future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1) if ts.index.inferred_freq == 'D' else ts.index[-1] + pd.Timedelta(weeks=1), 
                                 periods=forecast_periods, 
                                 freq=ts.index.inferred_freq)
    
    # 95% Confidence Interval is +/- 1.96 * std_error
    lower_ci = np.clip(forecast_mean.values - 1.96 * std_error, 0, None)
    upper_ci = forecast_mean.values + 1.96 * std_error
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast_mean.values,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci
    })
    
    metrics = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}
    return forecast_df, metrics, test_pred

def generate_demand_forecast(df, period_type='Daily', forecast_days=30):
    """
    Main entry point for forecasting.
    Aggregates data, fits ARIMA and Prophet/Holt-Winters, compares, 
    and returns the best forecast along with model comparisons.
    """
    logger.info("Starting demand forecasting for %s horizon (%d periods)...", period_type, forecast_days)
    ts = prepare_time_series(df, period_type)
    
    if len(ts) < 15:
        raise ValueError(f"Time series too short. Need at least 15 points, got {len(ts)}.")
        
    # Split point: 80/20 train/test
    train_len = int(len(ts) * 0.8)
    
    # Run ARIMA
    try:
        arima_forecast, arima_metrics, arima_pred = run_arima_forecast(ts, train_len, forecast_days)
    except Exception as e:
        logger.error("ARIMA forecasting failed: %s", str(e))
        arima_forecast, arima_metrics = None, {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')}
        
    # Run Prophet / Holt-Winters
    try:
        prophet_forecast, prophet_metrics, prophet_pred = run_prophet_forecast(ts, train_len, forecast_days)
    except Exception as e:
        logger.error("Alternative forecasting failed: %s", str(e))
        prophet_forecast, prophet_metrics = None, {'MAE': float('inf'), 'RMSE': float('inf'), 'MAPE': float('inf')}
        
    # Compare models based on MAPE (fallback to MAE if MAPE is 0)
    models_comparison = {
        'ARIMA': arima_metrics,
        'Prophet': prophet_metrics if PROPHET_AVAILABLE else {**prophet_metrics, 'Model': 'Holt-Winters (Fallback)'}
    }
    
    # Choose best model
    best_model_name = 'ARIMA'
    best_forecast = arima_forecast
    
    # Compare
    mape_arima = arima_metrics.get('MAPE', float('inf'))
    mape_prophet = prophet_metrics.get('MAPE', float('inf'))
    
    if mape_prophet < mape_arima and prophet_forecast is not None:
        best_model_name = 'Prophet' if PROPHET_AVAILABLE else 'Holt-Winters'
        best_forecast = prophet_forecast
        
    logger.info("Best model selected: %s with MAPE: %s%%", best_model_name, min(mape_arima, mape_prophet))
    
    # Format results
    comparison_df = pd.DataFrame(models_comparison).T.reset_index()
    comparison_df.rename(columns={'index': 'Model'}, inplace=True)
    
    return {
        'best_model': best_model_name,
        'forecast': best_forecast,
        'comparison': comparison_df,
        'historical': ts.reset_index()
    }
