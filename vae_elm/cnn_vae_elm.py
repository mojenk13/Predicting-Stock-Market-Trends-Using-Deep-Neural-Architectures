import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.dates as mdates
from data_loader import X_train, y_train, X_test, y_test, close_scaler

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_np = y_train.squeeze()
y_test_np = y_test.squeeze()

class CNNVAE(nn.Module):
    def __init__(self):
        super(CNNVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_mu = nn.Linear(64, 64)
        self.fc_logvar = nn.Linear(64, 64)
        self.decoder_fc = nn.Linear(64, 64)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 75) 
        )

    def encode(self, x):
        x = x.permute(0, 2, 1) 
        h = self.encoder(x).squeeze(-1)  
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        out = self.decoder(h)
        return out.view(-1, 15, 5)  

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

vae = CNNVAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld

print("Training CNN + VAE...")
vae.train()
for epoch in range(15):
    optimizer.zero_grad()
    recon_x, mu, logvar = vae(X_train_tensor)
    print("recon_x:", recon_x.shape, "X_train_tensor:", X_train_tensor.shape) 
    loss = vae_loss(recon_x, X_train_tensor, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

vae.eval()
with torch.no_grad():
    _, mu_train, _ = vae(X_train_tensor)
    _, mu_test, _ = vae(X_test_tensor)
    H_train = mu_train.numpy()
    H_test = mu_test.numpy()

def rbf_kernel(X, C, gamma=1.0):
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    C_norm = np.sum(C ** 2, axis=1).reshape(1, -1)
    dists = X_norm + C_norm - 2 * X @ C.T
    return np.exp(-gamma * dists)

np.random.seed(42)
num_centers = 50
center_idx = np.random.choice(len(H_train), num_centers, replace=False)
C = H_train[center_idx]

Z_train = rbf_kernel(H_train, C)
Z_test = rbf_kernel(H_test, C)

beta = np.linalg.pinv(Z_train) @ y_train_np
y_pred_norm = Z_test @ beta
y_pred = close_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
y_true = close_scaler.inverse_transform(y_test_np.reshape(-1, 1))

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nCNN + VAE + RBF ELM MAE: {mae:.2f}")
print(f"CNN + VAE + RBF ELM RMSE: {rmse:.2f}")

df = pd.read_csv("AAPL_data.csv", parse_dates=["Date"])
dates = df["Date"].values[-100:]

plt.figure(figsize=(10, 6))
plt.plot(dates, y_true, label="Actual Close")
plt.plot(dates, y_pred, label="Predicted Close")
plt.title("CNN + VAE + RBF ELM")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.decomposition import PCA
reduced = PCA(n_components=2).fit_transform(H_test)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y_test_np, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Normalized Close Price")
plt.title("Latent Space")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
plt.show()
