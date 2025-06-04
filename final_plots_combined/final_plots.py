import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cnn_vae import get_cnn_vae_predictions
from vae_elm import get_vae_elm_predictions
from train_predict import get_rbf_predictions

dates, actual, pred1 = get_cnn_vae_predictions()
_, _, pred2 = get_vae_elm_predictions()
_, _, pred3 = get_rbf_predictions()

dates = np.array(dates)
actual = np.array(actual)
pred1 = np.array(pred1)
pred2 = np.array(pred2)
pred3 = np.array(pred3)

plt.figure(figsize=(12, 6))
plt.plot(dates, actual, label="Actual", linewidth=2, color="black")
plt.plot(dates, pred1, label="CNN + VAE", linewidth=2, color="dodgerblue")
plt.plot(dates, pred2, label="VAE + ELM", linewidth=2, color="darkorange")
plt.plot(dates, pred3, label="RBF ELM", linewidth=2, color="green")

plt.scatter(dates, pred1, color="dodgerblue", s=20, label=None)
plt.scatter(dates, pred2, color="darkorange", s=20, label=None)
plt.scatter(dates, pred3, color="green", s=20, label=None)


plt.title("Combined Model Predictions", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=30)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
