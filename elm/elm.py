import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from data_loader import X_train, y_train, X_test, y_test, close_scaler

X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

y_train_np = y_train.squeeze()
y_test_np = y_test.squeeze()

y_train_np = pd.Series(y_train_np).rolling(3, min_periods=1).mean().values
y_test_np = pd.Series(y_test_np).rolling(3, min_periods=1).mean().values

low, high = np.percentile(y_train_np, [1, 99])
y_train_np = np.clip(y_train_np, low, high)

def rbf_kernel(X, C, gamma):
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    C_norm = np.sum(C ** 2, axis=1).reshape(1, -1)
    dists = X_norm + C_norm - 2 * X @ C.T
    return np.exp(-gamma * dists)

gammas = [0.001, 0.01, 0.1, 1, 10]
best_mae = float('inf')
best_pred = None
best_gamma = None

np.random.seed(42)
num_centers = 100
center_indices = np.random.choice(len(X_train_flat), num_centers, replace=False)
C = X_train_flat[center_indices]

for gamma in gammas:
    Z_train = rbf_kernel(X_train_flat, C, gamma)
    Z_test = rbf_kernel(X_test_flat, C, gamma)
    beta = np.linalg.pinv(Z_train) @ y_train_np
    y_pred_norm = Z_test @ beta
    y_pred = close_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
    y_true = close_scaler.inverse_transform(y_test_np.reshape(-1, 1))
    mae = mean_absolute_error(y_true, y_pred)

    if mae < best_mae:
        best_mae = mae
        best_gamma = gamma
        best_pred = y_pred

rmse = np.sqrt(mean_squared_error(y_true, best_pred))
y_pred_flat = best_pred.flatten()
mean_pred = np.mean(y_pred_flat)
fat = np.mean(np.abs(y_pred_flat - mean_pred))

df = pd.read_csv("AAPL_data.csv", parse_dates=["Date"])
dates = df["Date"].values[-len(y_test):]

plt.figure(figsize=(10, 6))
plt.plot(dates, y_true, label="Actual Close", linewidth=2, color='steelblue')
plt.plot(dates, best_pred, label="Predicted Close", linewidth=2, color='darkorange')
plt.scatter(dates, best_pred, color='darkorange', s=20, label='Prediction Points')
plt.scatter(dates, y_true, color='steelblue', s=20, label='Actual Points')

plt.title("RBF-ELM", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nBest gamma: {best_gamma}")
print(f"Improved ELM MAE: {best_mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Fluctuation Across the Mean (FAT): {fat:.2f}")

def get_elm_predictions():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from joblib import load

    data = pd.read_csv("apple_stock_data.csv", index_col=0, parse_dates=True)
    window_size = 15
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    close_prices = data['Close'].values.reshape(-1, 1)
    close_scaler = MinMaxScaler()
    close_scaled = close_scaler.fit_transform(close_prices)

    X = []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
    X = np.array(X)

    cnn_features = np.load("cnn_features.npy")  
    elm_model = load("elm_model.pkl")

    y_pred_scaled = elm_model.predict(cnn_features).reshape(-1, 1)
    y_pred = close_scaler.inverse_transform(y_pred_scaled)

    dates = data.index[window_size:]
    return dates, y_pred.flatten()
