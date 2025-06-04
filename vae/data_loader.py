import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


filename = "/Users/mollyjenkins/Desktop/final_project/group2/molly_code/vae/AAPL_data.csv"
stock_data = pd.read_csv(filename, parse_dates=['Date'])
stock_data.set_index('Date', inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = stock_data[features].copy()
data = data.apply(pd.to_numeric, errors='coerce').dropna()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

close_prices = data['Close'].values.reshape(-1, 1)
close_scaler = MinMaxScaler()
close_scaled = close_scaler.fit_transform(close_prices)

input_window = 15
X, y = [], []
for i in range(len(data_scaled) - input_window - 1):  
    X.append(data_scaled[i:i+input_window])
    y.append(close_scaled[i+input_window])  

X = np.array(X)  
y = np.array(y).reshape(-1, 1) 

X_train = X[:-100]
y_train = y[:-100]
X_test = X[-100:]
y_test = y[-100:]


