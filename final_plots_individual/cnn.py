import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

def feature_extraction_model():
    model = Sequential([
        Input(shape=(15, 5)),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=1),
        Flatten()
    ])
    print("Feature extraction model created.")
    return model

def get_cnn_predictions():
    data = pd.read_csv("AAPL_data.csv", index_col=0, parse_dates=True, skiprows=[1])
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features].apply(pd.to_numeric, errors='coerce').dropna()

    close_prices = data['Close'].values.reshape(-1, 1)
    close_scaler = MinMaxScaler()
    close_scaled = close_scaler.fit_transform(close_prices)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    window_size = 15
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i + window_size])
        y.append(close_scaled[i + window_size])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)


    X_train, y_train = X[:-100], y[:-100]
    X_test, y_test = X[-100:], y[-100:]
    test_dates = data.index[-100:]

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train.squeeze(), epochs=15, batch_size=32, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = close_scaler.inverse_transform(y_pred_scaled)
    y_true = close_scaler.inverse_transform(y_test)

    fat = np.mean(np.abs(y_pred.flatten() - np.mean(y_true.flatten())))
    print(f"Fluctuation Across the Mean (FAT): {fat:.4f}")

    return test_dates, y_pred.flatten()

if __name__ == "__main__":
    dates, y_pred = get_cnn_predictions()

    df = pd.read_csv("AAPL_data.csv", index_col=0, parse_dates=True)
    df = df[['Close']].apply(pd.to_numeric, errors='coerce').dropna()
    actual = df['Close'].iloc[-100:].values

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Close Prices', linewidth=2)
    plt.plot(dates, y_pred, label='Predicted Close Prices', linewidth=2, color='orange')
    plt.scatter(dates, y_pred, color='orange', s=20, label='Prediction Points')
    plt.scatter(dates, actual, color='blue', s=20, label='Actual Points')

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
