import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

input_window = 15
X_all, y_all = create_sliding_sequences(scaled_data, input_window)
X_train, y_train = X_all[:-100], y_all[:-100]
X_test, y_test = X_all[-100:], y_all[-100:]

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1)

class StockForecastDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(StockForecastDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(StockForecastDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

class VAEForecast(nn.Module):
    def __init__(self, input_dim=5, seq_len=15, latent_dim=32, hidden_dim=128):
        super(VAEForecast, self).__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

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
        h_dec = self.latent_to_hidden(z)
        out = self.decoder(h_dec)
        return out, mu, logvar

def vae_loss(recon, target, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def train_vae(model, dataloader, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        beta = min(1.0, epoch / 50)
        total_loss, total_recon, total_kl = 0, 0, 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, mu, logvar = model(batch_x)
            loss, recon, kl = vae_loss(output, batch_y, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, β={beta:.2f}, Loss: {total_loss:.4f}, Recon: {total_recon:.4f}, KL: {total_kl:.4f}")

def evaluate_vae(model, dataloader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output, _, _ = model(batch_x)
            preds.extend(output.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    return np.array(preds), np.array(actuals)

def plot_forecast(pred, actual, reference_dates, close_mean, close_std):
    pred_close = pred[:, 0] * close_std + close_mean
    actual_close = actual[:, 0] * close_std + close_mean

    plt.figure(figsize=(12, 6))
    plt.plot(reference_dates, actual_close, label='Actual Close', linewidth=2)
    plt.plot(reference_dates, pred_close, label='Predicted Close', linewidth=2, color='orange')
    plt.scatter(reference_dates, pred_close, color='orange', s=20, label='Prediction Points')
    plt.scatter(reference_dates, actual_close, color='blue', s=20, label='Actual Points')

    plt.title('VAE')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    rmse = np.sqrt(np.mean((actual_close - pred_close) ** 2))
    print(f"RMSE: ${rmse:.2f}")

model = VAEForecast(input_dim=5, seq_len=15, latent_dim=32, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_vae(model, train_loader, optimizer, num_epochs=20)
preds, actuals = evaluate_vae(model, test_loader)

pred_close = preds[:, 0] * close_std + close_mean
mean_pred = np.mean(pred_close)
fat = np.mean(np.abs(pred_close - mean_pred))
print(f"Fluctuation Across the Mean (FAT): {fat:.2f}")

actual_close = actuals[:, 0] * close_std + close_mean
mae = np.mean(np.abs(actual_close - pred_close))
print(f"Mean Absolute Error (MAE): {mae:.2f}")

test_dates = apple_data["Date"].iloc[-100:].values
plot_forecast(preds, actuals, test_dates, close_mean, close_std)

def get_vae_predictions():
    import pandas as pd
    import numpy as np
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler

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

    vae_model = load_model("vae_model.h5")  
    X_pred = vae_model.predict(X)

    y_pred_scaled = X_pred[:, -1, 3].reshape(-1, 1)  
    y_pred = close_scaler.inverse_transform(y_pred_scaled)

    dates = data.index[window_size:]
    return dates, y_pred.flatten()



