import yfinance as yf
import pandas as pd
import os

ticker = "^GSPC"
start_date = "1928-01-01"
end_date = "2025-06-01"

stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.columns.name = None  

output_folder = "/Users/mollyjenkins/Desktop/final_project/group2/molly_code/historical_comparison"
os.makedirs(output_folder, exist_ok=True)

stock_data.reset_index().to_csv(f"{output_folder}/s&p500_data.csv", index=False)
print("Saved s&p500 CSV successfully.")
