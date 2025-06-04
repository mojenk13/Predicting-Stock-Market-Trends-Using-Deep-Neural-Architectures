import yfinance as yf
import pandas as pd

ticker = "AAPL"
start_date = "2019-01-01"
end_date = "2025-06-04"

stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.columns.name = None
stock_data.reset_index().to_csv("/Users/mollyjenkins/Desktop/final_project/group2/molly_code/predicting_future_cnnelm/AAPL_data.csv", index=False)

print("saved clean CSV to final_project folder.")
