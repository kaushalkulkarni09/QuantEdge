# download_data.py
# This script is designed to be called from a C++ application.
# It downloads historical stock data using yfinance and saves it to a CSV.

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

def download_and_save_data(ticker, start_date, end_date, output_csv_path):
    """
    Downloads historical data for a given ticker and date range, calculates log returns,
    and saves the 'Log Returns' column to a specified CSV path.

    Args:
        ticker (str): Stock ticker symbol (e.g., "SPY", "AAPL").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        output_csv_path (str): The full path where the processed CSV will be saved.

    Returns:
        bool: True if data was successfully downloaded and saved, False otherwise.
    """
    try:
        # When run from C++, yfinance.download output needs to be suppressed

        if __name__ != '__main__': # Only redirect if not running directly
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w') # Redirect stdout to null device

        print(f"Python: Attempting to download data for {ticker} from {start_date} to {end_date}...", file=sys.stderr)

        # Download data, group_by=False prevents MultiIndex for single ticker
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", group_by=False)

        if __name__ != '__main__':
            sys.stdout = old_stdout


        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
            print(f"Python: MultiIndex columns flattened. New columns: {data.columns.tolist()}", file=sys.stderr)

        print(f"Python: Download complete. DataFrame shape: {data.shape}", file=sys.stderr)
        print(f"Python: DataFrame columns: {data.columns.tolist()}", file=sys.stderr)
        print(f"Python: DataFrame head:\n{data.head()}", file=sys.stderr)

        if data.empty:
            print(f"PythonDataFetcher: No data downloaded for {ticker}. Check ticker symbol and date range.", file=sys.stderr)
            return False

        # Use 'Adj Close' primarily, fallback to 'Close'
        target_column = None
        if 'Adj Close' in data.columns:
            target_column = 'Adj Close'
        elif 'Close' in data.columns:
            target_column = 'Close'
        else:
            print(f"PythonDataFetcher: Error: Neither 'Adj Close' nor 'Close' column found in downloaded data.", file=sys.stderr)
            return False

        # Calculate Daily Log Returns using the identified target_column
        data[target_column] = pd.to_numeric(data[target_column], errors='coerce') # Ensure numeric
        data['Log Returns'] = np.log(data[target_column] / data[target_column].shift(1))

        # Drop rows with NaN values in 'Log Returns' (e.g., first row, or any missing data points)
        data = data.dropna(subset=['Log Returns'])

        # --- NEW FIX: Ensure all 'Log Returns' are numeric and fill any remaining NaNs ---
        data['Log Returns'] = pd.to_numeric(data['Log Returns'], errors='coerce') # Re-coerce after dropna for safety
        data['Log Returns'].fillna(0.0, inplace=True) # Fill any remaining NaNs with 0.0
        # --- END NEW FIX ---

        if data.empty:
            print(f"PythonDataFetcher: No valid log returns calculated after dropping NaNs. Data might be too short or all NaNs.", file=sys.stderr)
            return False

        # Select only the 'Log Returns' column and save to CSV
        output_df = data[['Log Returns']]

        # Debug print statements (already there)
        print(f"Python debug: Current working directory is: {os.getcwd()}", file=sys.stderr)
        print(f"Python debug: Attempting to save CSV to: {os.path.abspath(output_csv_path)}", file=sys.stderr)

        # Ensure to_csv arguments for clean output, including precision
        output_df.to_csv(output_csv_path, index=False, header=True, float_format='%.10f')

        print(f"PythonDataFetcher: Successfully processed data for {ticker}. Log returns saved to {output_csv_path}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"PythonDataFetcher Error: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':

    print("Running download_data.py directly for testing/debugging...")
    if len(sys.argv) == 5:
        ticker_arg = sys.argv[1]
        start_date_arg = sys.argv[2]
        end_date_arg = sys.argv[3]
        output_csv_arg = sys.argv[4]
        success = download_and_save_data(ticker_arg, start_date_arg, end_date_arg, output_csv_arg)
    else:
        print("Usage: python download_data.py <ticker> <start_date> <end_date> <output_csv_path>")
        print("Example: python download_data.py AAPL 2020-01-01 2024-12-31 test_output.csv")
        success = False # Indicate failure due to incorrect arguments

    if success:
        print("Test download successful.")
    else:
        print("Test download failed.")