import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, GlobalAveragePooling1D
import joblib

df = pd.read_csv("AAPL_data.csv", index_col=0, parse_dates=True)

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

window_size = 15
dates = df.index[-100:]

scaler = MinMaxScaler()
data = scaler.fit_transform(df)

close_prices = df['Close'].values.reshape(-1, 1)
close_scaler = MinMaxScaler()
close_scaled = close_scaler.fit_transform(close_prices)

X, y = [], []
for i in range(len(data) - window_size):
    X.append(data[i:i + window_size])
    y.append(close_scaled[i + window_size])
X = np.array(X)
y = np.array(y).reshape(-1, 1)

X_train, y_train = X[:-100], y[:-100]
X_test, y_test = X[-100:], y[-100:]

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
joblib.dump(close_scaler, "close_scaler.pkl")

def cnn_prediction():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train.squeeze(), epochs=15, batch_size=32)

    predictions = model.predict(X_test)
    predictions_rescaled = close_scaler.inverse_transform(predictions)
    y_test_rescaled = close_scaler.inverse_transform(y_test)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test_rescaled, label='Actual Close Prices', linewidth=2)
    plt.plot(dates, predictions_rescaled, label='Predicted Close Prices', linewidth=2, color='orange')
    plt.scatter(dates, predictions_rescaled, color='orange', s=20, label='Prediction Points')
    plt.scatter(dates, y_test_rescaled, color='blue', s=20, label='Actual Points')

    plt.title("CNN")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    np.save("cnn_predictions.npy", predictions_rescaled)

    fat = np.mean(np.abs(predictions_rescaled.flatten() - np.mean(y_test_rescaled.flatten())))
    print(f"Fluctuation Across the Mean (FAT): {fat:.4f}")

    return model

def feature_extraction_model():
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=1),
        GlobalAveragePooling1D()
    ])
    print("Feature extraction mode" \
    "l created.")
    return model

def get_cnn_predictions():
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import load_model
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

    model = load_model("cnn_model.h5")  
    y_pred_scaled = model.predict(X)
    y_pred = close_scaler.inverse_transform(y_pred_scaled)

    dates = data.index[window_size:]
    return dates, y_pred.flatten()



if __name__ == "__main__":
    mode = input("Choose mode: 'predict' or 'filter': ").strip().lower()

    if mode == "predict":
        cnn_prediction()
    elif mode == "filter":
        feature_extraction_model()
    else:
        print("Invalid. Choose 'predict' or 'filter'.")

        
