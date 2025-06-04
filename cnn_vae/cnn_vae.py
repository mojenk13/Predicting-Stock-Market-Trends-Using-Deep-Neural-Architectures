import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("AAPL_data.csv", parse_dates=["Date"])
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
X_seq, y_seq = [], []
for i in range(len(scaled_data) - window_size):
    X_seq.append(scaled_data[i:i + window_size])
    y_seq.append(close_scaled[i + window_size])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_train, X_test = X_seq[:-100], X_seq[-100:]
y_train, y_test = y_seq[:-100], y_seq[-100:]

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

class CNNVAEForecaster(nn.Module):
    def __init__(self, input_dim=5, latent_dim=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3),  
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.forecast_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)
        h = self.cnn(x).squeeze(-1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.forecast_head(z)
        return pred, mu, logvar

model = CNNVAEForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5) 

def loss_fn(pred, true, mu, logvar, beta=0.0005):  
    recon = F.mse_loss(pred, true)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld

for epoch in range(500): 
    model.train()
    pred, mu, logvar = model(X_train_tensor)
    loss = loss_fn(pred, y_train_tensor, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred_test, _, _ = model(X_test_tensor)
    y_pred = pred_test.numpy()
    y_true = y_test_tensor.numpy()

y_pred_inv = close_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_true_inv = close_scaler.inverse_transform(y_true.reshape(-1, 1))

dates = df['Date'].values[-100:]
plt.figure(figsize=(10, 6))
plt.plot(dates, y_true_inv, label="Actual Close", linewidth=2, color='blue', marker='o', markersize=3)
plt.plot(dates, y_pred_inv, label="Predicted Close", linewidth=2, color='orange', marker='o', markersize=3)

plt.title("CNN + VAE Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


mae = mean_absolute_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
fat = np.mean(np.abs(y_pred_inv.flatten() - np.mean(y_pred_inv)))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"FAT: {fat:.2f}")

def get_cnn_vae_predictions():
    return df['Date'].values[-100:], y_true_inv.flatten(), y_pred_inv.flatten()
