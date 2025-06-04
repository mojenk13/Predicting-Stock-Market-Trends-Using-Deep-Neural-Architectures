import pandas as pd
import matplotlib.pyplot as plt
from cnn import get_cnn_predictions
from vae import get_vae_predictions
from elm import get_elm_predictions

dates, cnn_preds = get_cnn_predictions()
_, vae_preds = get_vae_predictions()
_, elm_preds = get_elm_predictions()


df = pd.read_csv("AAPL_data.csv", index_col=0, parse_dates=True, skiprows=[1])
df = df[['Close']].apply(pd.to_numeric, errors='coerce').dropna()
actual_close = df['Close'].iloc[-100:].values  

plt.figure(figsize=(12, 6))
plt.plot(dates, actual_close, label="Actual", linewidth=2, color="black")
plt.plot(dates, cnn_preds, label="CNN", linewidth=2, color="dodgerblue")
plt.plot(dates, vae_preds, label="VAE", linewidth=2, color="darkorange")
plt.plot(dates, elm_preds, label="ELM", linewidth=2, color="green")

plt.scatter(dates, cnn_preds, s=20, color="dodgerblue", label=None)
plt.scatter(dates, vae_preds, s=20, color="darkorange", label=None)
plt.scatter(dates, elm_preds, s=20, color="green", label=None)


plt.title("CNN, VAE, and ELM", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
