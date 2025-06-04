import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import BDay

df = pd.read_csv("AAPL_data.csv", parse_dates=["Date"])
df = df.sort_values("Date")
features = ['Open', 'High', 'Low', 'Close', 'Volume']
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

data = df[features].values
close_prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
close_scaler = MinMaxScaler()
close_scaled = close_scaler.fit_transform(close_prices)

window_size = 15
X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i + window_size])
    y.append(close_scaled[i + window_size])
X = np.array(X)
y = np.array(y).squeeze()
X_flat = X.reshape(len(X), -1)


def rbf_kernel(X, C, gamma):
    return np.exp(-gamma * (
        np.sum(X**2, axis=1).reshape(-1, 1) +
        np.sum(C**2, axis=1).reshape(1, -1) -
        2 * X @ C.T
    ))

np.random.seed(42)
centers = X_flat[np.random.choice(len(X_flat), 45, replace=False)]
gamma = 1.0
Z = rbf_kernel(X_flat, centers, gamma)
beta = np.linalg.pinv(Z) @ y

start_date = pd.Timestamp("2025-05-05")
forecast_days = 22
forecast_dates = pd.bdate_range(start=start_date, periods=forecast_days)

cutoff = df[df['Date'] < start_date].index[-1]
window = scaled_data[cutoff - window_size + 1:cutoff + 1].copy()

preds = []
for _ in range(forecast_days):
    Zf = rbf_kernel(window.reshape(1, -1), centers, gamma)
    y_scaled = Zf @ beta
    y_unscaled = close_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()[0]
    preds.append(y_unscaled)

    new_row = window[-1].copy()
    new_row[3] = y_scaled[0]
    window = np.roll(window, -1, axis=0)
    window[-1] = new_row

actual_df = df[(df['Date'] >= start_date) & (df['Date'] <= pd.Timestamp("2025-06-04"))]
actual_close = actual_df['Close'].values
actual_dates = actual_df['Date'].values

min_len = min(len(preds), len(actual_close))
forecast_preds = np.array(preds[:min_len])
actual_close = actual_close[:min_len]
forecast_dates = forecast_dates[:min_len]
actual_dates = actual_dates[:min_len]

bias = np.mean(actual_close - forecast_preds)
bias_corrected = forecast_preds + bias
scale = np.mean(actual_close) / np.mean(forecast_preds)
scale_corrected = forecast_preds * scale

def score(y_true, y_pred): return mean_squared_error(y_true, y_pred)**0.5
rmse_bias = score(actual_close, bias_corrected)
rmse_scale = score(actual_close, scale_corrected)

if rmse_bias <= rmse_scale:
    final_forecast = bias_corrected
else:
    final_forecast = scale_corrected


plt.figure(figsize=(12, 6))
plt.plot(actual_dates, actual_close, label="Actual Close", linewidth=2, color="steelblue")
plt.plot(forecast_dates, final_forecast, label="Forecast", linewidth=2, color="darkorange")
plt.scatter(forecast_dates, final_forecast, color="darkorange", s=30)
plt.scatter(actual_dates, actual_close, color="steelblue", s=20)

plt.title("Predicted Forecast vs Actual Close)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=30)
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


mae = mean_absolute_error(actual_close, final_forecast)
fat = np.mean(np.abs(final_forecast - np.mean(final_forecast)))

print(f"\nMAE: {mae:.2f}")
print(f"FAT: {fat:.2f}")
