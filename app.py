import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import io

# --- Embedded CSV Data ---
# The data from the provided CSV files is embedded here as strings.
# This makes the script self-contained and easy to run.

AAPL_DATA = """
Date,Open,High,Low,Close,Adj Close,Volume
1998-11-30,1.234375,1.243304,1.133929,1.140625,0.763834,140372400
1998-12-01,1.142857,1.243304,1.129464,1.218750,0.816152,216434400
1998-12-02,1.218750,1.316964,1.196429,1.285714,0.860995,240620800
1998-12-03,1.296875,1.303571,1.200893,1.203125,0.805688,156511600
1998-12-04,1.225446,1.229911,1.142857,1.169643,0.783266,180342400
2018-10-31,216.880005,220.449997,216.619995,218.860001,218.099014,38358900
2018-11-01,219.050003,222.360001,216.809998,222.220001,221.447327,58323200
2018-11-02,209.550003,213.649994,205.429993,207.479996,206.758575,91328700
2018-11-05,204.300003,204.389999,198.169998,201.589996,200.889053,66163700
2018-11-06,201.919998,204.720001,201.690002,203.770004,203.061493,31882900
"""

AMZN_DATA = """
Date,Open,High,Low,Close,Adj Close,Volume
1998-11-30,36.604168,36.708332,31.958334,32.000000,32.000000,29752200
1998-12-01,30.916666,34.958332,30.333334,34.916668,34.916668,47154600
1998-12-02,34.489582,34.583332,32.927082,33.250000,33.250000,29649000
1998-12-03,33.187500,34.083332,30.833334,31.583334,31.583334,21836400
1998-12-04,32.583332,32.833332,30.416666,31.416666,31.416666,22257600
2018-10-31,1570.579956,1598.319946,1537.000000,1598.010010,1598.010010,7986800
2018-11-01,1623.530029,1670.449951,1598.439941,1665.530029,1665.530029,8135500
2018-11-02,1678.589966,1697.439941,1651.829956,1665.530029,1665.530029,6955500
2018-11-05,1657.569946,1658.089966,1596.359985,1627.800049,1627.800049,5624700
2018-11-06,1618.349976,1665.000000,1614.550049,1642.810059,1642.810059,4257400
"""

FB_DATA = """
Date,Open,High,Low,Close,Adj Close,Volume
2012-05-18,42.049999,45.000000,38.000000,38.230000,38.230000,573576400
2012-05-21,36.529999,36.660000,33.000000,34.029999,34.029999,168192700
2012-05-22,32.610001,33.590000,30.940001,31.000000,31.000000,101786600
2012-05-23,31.370001,32.500000,31.360001,32.000000,32.000000,73600000
2012-05-24,32.950001,33.209999,31.770000,33.029999,33.029999,50237200
2018-10-31,155.000000,156.399994,148.960007,151.789993,151.789993,60101300
2018-11-01,151.520004,152.750000,149.350006,151.750000,151.750000,25640800
2018-11-02,151.800003,154.130005,148.960007,150.350006,150.350006,24708700
2018-11-05,150.100006,150.190002,147.440002,148.679993,148.679993,15971200
2018-11-06,149.309998,150.970001,148.000000,149.940002,149.940002,16667100
"""

# Dictionary to map ticker symbols to their corresponding data strings
STOCK_DATA = {
    "AAPL": AAPL_DATA,
    "AMZN": AMZN_DATA,
    "FB": FB_DATA
}

# --- App Title and Description ---
st.set_page_config(layout="wide")
st.title("Time Series Stock Market Analysis")
st.write("""
This application provides a comprehensive analysis of stock market data using various forecasting models.
Select a stock ticker from the sidebar, choose a date range, and pick a model to generate a forecast.
The data for this app is sourced from local CSV files.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("User Input for Forecasting")

ticker_symbol = st.sidebar.selectbox(
    "Select Stock Ticker",
    ("AAPL", "AMZN", "FB")
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- Data Loading and Display ---
@st.cache_data
def load_data(ticker):
    """
    Loads data for the selected ticker from the embedded string data.
    Uses io.StringIO to read the string as if it were a file.
    """
    csv_data = STOCK_DATA[ticker]
    data = pd.read_csv(io.StringIO(csv_data), index_col='Date', parse_dates=True)
    # Ensure the 'Close' column is numeric, coercing errors to NaN
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    # Remove rows with NaN in 'Close' column
    data.dropna(subset=['Close'], inplace=True)
    return data

data_load_state = st.text("Loading data...")
full_data = load_data(ticker_symbol)
data = full_data.loc[start_date:end_date]
data_load_state.text(f"Loading data for {ticker_symbol}... done!")

st.subheader("Raw Data for Forecasting")
st.write("Displaying the latest 5 records from the selected date range.")
st.write(data.tail())

# --- Plotting Historical Data ---
st.subheader("Stock Closing Price Over Time (for Forecasting)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'])
ax.set_title(f"{ticker_symbol} Stock Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (USD)")
ax.grid(True)
st.pyplot(fig)

# --- Model Selection and Forecasting ---
st.sidebar.header("Forecasting Model")
model_selection = st.sidebar.selectbox(
    "Choose a model",
    ["Prophet", "ARIMA", "SARIMA", "LSTM"]
)

if st.sidebar.button("Generate Forecast"):
    if data.empty:
        st.warning("No data available for the selected date range. Please select a different range.")
    else:
        st.subheader(f"Forecast with {model_selection}")
        forecast_state = st.text(f"Generating {model_selection} forecast...")

        if model_selection == "Prophet":
            # --- FIX START: Robust data preparation for Prophet ---
            # This block prepares the data in the exact format Prophet needs,
            # preventing the TypeError.
            df_prophet = data[['Close']].reset_index().rename(
                columns={'Date': 'ds', 'Close': 'y'}
            )
            
            df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
            df_prophet.dropna(subset=['y'], inplace=True)

            if df_prophet.empty:
                st.warning("Not enough valid data for Prophet model in the selected date range.")
            else:
                model_prophet = Prophet()
                model_prophet.fit(df_prophet)
                future = model_prophet.make_future_dataframe(periods=365)
                forecast_prophet = model_prophet.predict(future)
                st.write(f'### {model_selection} Forecast Plot')
                fig_prophet = model_prophet.plot(forecast_prophet)
                st.pyplot(fig_prophet)
                st.write(f'### {model_selection} Forecast Components')
                fig_components = model_prophet.plot_components(forecast_prophet)
                st.pyplot(fig_components)
            # --- FIX END ---

        elif model_selection == "ARIMA":
            model_arima = ARIMA(data['Close'], order=(5, 1, 0))
            model_fit_arima = model_arima.fit()
            forecast_arima = model_fit_arima.forecast(steps=30)
            fig_arima, ax_arima = plt.subplots(figsize=(12, 6))
            ax_arima.plot(data['Close'], label='Actual')
            ax_arima.plot(forecast_arima, label='ARIMA Forecast', color='red')
            ax_arima.set_title('ARIMA Forecast')
            ax_arima.legend()
            st.pyplot(fig_arima)

        elif model_selection == "SARIMA":
            # The seasonal_order 'm' should be appropriate for the data frequency
            # Using m=12 for monthly patterns, adjust if your data is different
            model_sarima = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit_sarima = model_sarima.fit(disp=False)
            forecast_sarima_res = model_fit_sarima.get_forecast(steps=30)
            forecast_ci = forecast_sarima_res.conf_int()
            fig_sarima, ax_sarima = plt.subplots(figsize=(12, 6))
            ax_sarima.plot(data['Close'], label='Actual')
            forecast_sarima_res.predicted_mean.plot(label='SARIMA Forecast', color='green', ax=ax_sarima)
            ax_sarima.fill_between(forecast_ci.index,
                                   forecast_ci.iloc[:, 0],
                                   forecast_ci.iloc[:, 1], color='k', alpha=.15)
            ax_sarima.set_title('SARIMA Forecast')
            ax_sarima.legend()
            st.pyplot(fig_sarima)

        elif model_selection == "LSTM":
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data[['Close']].values)
            
            time_step = 60
            if len(scaled_data) <= time_step:
                st.warning("Not enough data for LSTM model. Need more than 60 data points.")
            else:
                # Prepare dataset for training
                X_train, y_train = [], []
                for i in range(time_step, len(scaled_data)):
                    X_train.append(scaled_data[i-time_step:i, 0])
                    y_train.append(scaled_data[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

                # Build and train the LSTM model
                model_lstm = Sequential()
                model_lstm.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model_lstm.add(LSTM(50, return_sequences=False))
                model_lstm.add(Dense(25))
                model_lstm.add(Dense(1))
                model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                model_lstm.fit(X_train, y_train, batch_size=1, epochs=1)

                # Forecasting future values
                last_60_days = scaled_data[-60:].reshape(1, -60, 1)
                future_predictions = []
                for _ in range(30): # Predict next 30 days
                    next_pred = model_lstm.predict(last_60_days)
                    future_predictions.append(next_pred[0,0])
                    # Update last_60_days to include the new prediction
                    last_60_days = np.append(last_60_days[:,1:,:], [[next_pred]], axis=1)

                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                
                # Plotting results
                fig_lstm, ax_lstm = plt.subplots(figsize=(12, 6))
                ax_lstm.plot(data.index, data['Close'], label='Actual Price')
                ax_lstm.plot(future_dates, future_predictions, label='LSTM Forecast', color='orange')
                ax_lstm.set_title('LSTM Forecast')
                ax_lstm.legend()
                st.pyplot(fig_lstm)

        forecast_state.text(f"Forecast with {model_selection} generated successfully!", help="Scroll down to see the results.")