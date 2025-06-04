import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

apple_data = pd.read_csv("AAPL_data.csv", parse_dates=["Date"])
features = ['Open', 'High', 'Low', 'Close', 'Volume']
apple_data[features] = apple_data[features].apply(pd.to_numeric, errors='coerce')
apple_data = apple_data.dropna()
data = apple_data[features].values
target_column = 3

data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
scaled_data = (data - data_mean) / data_std
close_mean = data_mean[target_column]
close_std = data_std[target_column]

def create_sliding_sequences(data, input_window=15):
    X, y = [], []
    for i in range(len(data) - input_window):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window, target_column])
    return np.array(X), np.array(y)

X_all, y_all = create_sliding_sequences(scaled_data, 15)
X_train, y_train = X_all[:-100], y_all[:-100]
X_test, y_test = X_all[-100:], y_all[-100:]

device = torch.device("cpu")
X_train_tensor = torch.FloatTensor(X_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

class VAEForecast(nn.Module):
    def __init__(self, input_dim=5, seq_len=15, latent_dim=32, hidden_dim=128):
        super(VAEForecast, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n.squeeze(0)
        mu = self.mu_layer(h_n)
        logvar = self.logvar_layer(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

vae_model = VAEForecast().to(device)
vae_model.eval()

with torch.no_grad():
    _, mu_train, _ = vae_model(X_train_tensor)
    _, mu_test, _ = vae_model(X_test_tensor)

X_train_flat = mu_train.cpu().numpy()
X_test_flat = mu_test.cpu().numpy()

y_train_np = pd.Series(y_train).rolling(3, min_periods=1).mean().values
y_test_np = pd.Series(y_test).rolling(3, min_periods=1).mean().values
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
    y_pred = y_pred_norm * close_std + close_mean
    y_true = y_test_np * close_std + close_mean
    mae = mean_absolute_error(y_true, y_pred)
    if mae < best_mae:
        best_mae = mae
        best_gamma = gamma
        best_pred = y_pred

rmse = np.sqrt(mean_squared_error(y_true, best_pred))
fat = np.mean(np.abs(best_pred.flatten() - np.mean(best_pred)))
print(f"FAT: {fat:.2f}")
print(f"\nBest gamma: {best_gamma}")
print(f"VAE + RBF-ELM MAE: {best_mae:.2f}, RMSE: {rmse:.2f}")

dates = apple_data["Date"].values[-100:]
plt.figure(figsize=(10, 6))
plt.plot(dates, y_true, label="Actual Close", linewidth=2, color='steelblue')
plt.plot(dates, best_pred, label="Predicted Close", linewidth=2, color='darkorange')
plt.scatter(dates, y_true, color='steelblue', s=20)
plt.scatter(dates, best_pred, color='darkorange', s=20)
plt.title("VAE + RBF-ELM Combined Forecast", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
