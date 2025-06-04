import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.manifold import TSNE
from torchinfo import summary
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from cnn_model import CNNEncoder
from data_loader import X_train, y_train, X_test, y_test, close_scaler

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_np = y_train.squeeze()
y_test_np = y_test.squeeze()

encoder = CNNEncoder(input_features=5, output_dim=64)

print("\nCNN  Summary:")
summary(encoder, input_size=(8, 15, 5)) 

print("\nData Shapes:")
print(f"X_train_tensor: {X_train_tensor.shape}")
print(f"X_test_tensor: {X_test_tensor.shape}")

with torch.no_grad():
    H_train = encoder(X_train_tensor).numpy()  
    H_test = encoder(X_test_tensor).numpy()

print(f"Encoded H_train: {H_train.shape}")
print(f"Encoded H_test: {H_test.shape}")

def rbf_kernel(X, C, gamma):
    """X: (N, D), C: (M, D)"""
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    C_norm = np.sum(C ** 2, axis=1).reshape(1, -1)
    dists = X_norm + C_norm - 2 * X @ C.T
    return np.exp(-gamma * dists)

np.random.seed(42)
num_centers = 150
center_indices = np.random.choice(len(H_train), num_centers, replace=False)
C = H_train[center_indices]
gamma = 1.0

Z_train = rbf_kernel(H_train, C, gamma)  
Z_test = rbf_kernel(H_test, C, gamma)

print("\nRBF Kernel Output:")
print(f"Z_train: {Z_train.shape}")
print(f"Z_test: {Z_test.shape}")

beta = np.linalg.pinv(Z_train) @ y_train_np  

y_pred_norm = Z_test @ beta
y_pred = close_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
y_true = close_scaler.inverse_transform(y_test_np.reshape(-1, 1))

print("\nPredictions:")
print(f"y_pred_norm (scaled): {y_pred_norm.shape}")
print(f"y_pred (actual close prices): {y_pred.shape}")

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"\nRBF ELM MAE: {mae:.2f}, RMSE: {rmse:.2f}")

df = pd.read_csv('s&p500_data.csv', parse_dates=["Date"])
test_days = len(y_test)  
dates = df["Date"].values[-test_days:]
plt.figure(figsize=(10, 6))
plt.plot(dates, y_true, label="Actual Close")
plt.plot(dates, y_pred, label="Predicted Close")

plt.title("S&P 500 with CNN + RBF ELM")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  
plt.xticks(rotation=30, ha='right')  

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nStart date of prediction:", dates[0])
print("End date of prediction:", dates[-1])

fat = np.mean(np.abs(y_pred.flatten() - np.mean(y_true.flatten())))
print(f"FAT: {fat:.2f}")

reduced = PCA(n_components=2).fit_transform(H_test)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y_test_np, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Normalized Close Price")
plt.title("Latent Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

fat = np.mean(np.abs(y_pred.flatten() - np.mean(y_true.flatten())))
print(f"FAT: {fat:.2f}")
