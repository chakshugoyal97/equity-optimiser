# save_data.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

def fetch_fresh():
    # Fetch top 500 tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500 = pd.read_html(url)[0]
    tickers = sp500['Symbol'].tolist()

    print(tickers)

    data = yf.download(tickers, period="3y", group_by='ticker', progress=True)    
    with open(DATA_DIR / "raw_spx.pickle", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data

def save_market_data(DATA_DIR):   
    # Download historical data (3 years)
    try:
        with open(DATA_DIR / "raw_spx.pickle", 'rb') as handle:
            data = pickle.load(handle)
    except:
        data = fetch_fresh()
    
    # Process data
    returns_dict, volumes_dict = {}, {}
    ticker_keys = []
    
    for ticker in data.columns.get_level_values(0).unique():
        daily_prices = data[ticker]['Close']

        daily_returns = daily_prices.pct_change(fill_method=None).iloc[1:]
        
        if daily_returns.isnull().values.any() or \
            np.any(daily_returns <= -1) or \
            np.any(daily_returns >= 1):
            print(f"ignoring ticker because of data issues: {ticker}")
            continue
        continue_flag = False
        for t in ticker_keys:
            corr = data[ticker]['Close'].corr(data[t]['Close'])
            # (GOOGL, GOOG); (NWS, NWSA); (FOX, FOXA); (CTAS, TT); (DHI, LEN); 
            if corr >= 0.99:
                print(f"high corr: {corr}, bw {t} and {ticker}, dropping {ticker}")
                continue_flag = True
                break
        if continue_flag:
            continue
        returns_dict[ticker] = daily_returns
        volumes_dict[ticker] = (data[ticker]['Volume'] * data[ticker]['Close']).mean()
        ticker_keys.append(ticker)

    # Create DataFrames with ticker information
    returns_df = pd.DataFrame(returns_dict)
    volumes = pd.Series(volumes_dict, name='ADV')
    
    # Calculate mean and covariance, roughly annualised by t=252
    # Note that this is if we assume that stock returns follow N(mu, sigma^2)
    # And that each ri is an iid.
    # For a stock, r1, r2, ..., rt are random variables that all follow N(mu, sigma^2)
    # r_ann is annualised return.
    # E(r_ann) = E(r_1 +...+ r_t) = E(r_1)+...+ E(r_t) = t*E(r_1) = t*mu
    # Var(r_ann) = Var(r_1 +...+ r_t) = Var(r_1)+...+ Var(r_t) = t*Var(r_1) = t*sigma^2 
    
    expected_returns = returns_df.mean() * 252
    covariance_matrix = returns_df.cov() * 252
    
    # 5. Save data with ticker metadata
    # Save tickers as list, expected returns, covariance and volume  
    pd.Series(ticker_keys, name='Ticker').to_csv(DATA_DIR / "sp500_tickers.csv")
    expected_returns.rename('ExpectedReturn').to_csv(DATA_DIR / "expected_returns.csv")
    covariance_matrix.to_csv(DATA_DIR / "covariance_matrix.csv")
    volumes.to_csv(DATA_DIR / "average_daily_volumes.csv")
    

if __name__ == "__main__":
    DATA_DIR = Path("./workbooks/spx_data")
    DATA_DIR.mkdir(exist_ok=True)
    save_market_data(DATA_DIR)
    print(f"Data saved to {DATA_DIR} directory")


def load_optimization_data(DATA_DIR, n: int):
    """
    Utility Function for Loading the saved data in save_market_data
    Args
        DATA_DIR: which directory is the data in
        n: number of stocks, max 500
    """

    # Load tickers
    tickers = pd.read_csv(DATA_DIR / "sp500_tickers.csv")['Ticker'].tolist()

    # Load expected returns with ticker index
    er_df = pd.read_csv(DATA_DIR / "expected_returns.csv", index_col=0)
    er = er_df['ExpectedReturn'].values.astype(np.float64)

    # Load covariance matrix with ticker index
    cov_df = pd.read_csv(DATA_DIR / "covariance_matrix.csv", index_col=0)
    cov = cov_df.values.astype(np.float64)

    # Load ADV with ticker index
    adv_df = pd.read_csv(DATA_DIR / "average_daily_volumes.csv", index_col=0)
    adv = adv_df['ADV'].values.astype(np.float64)

    assert len(tickers) == len(er) == len(adv) == cov.shape[0] == cov.shape[1], \
        "Mismatched dimensions between inputs"
    return {
        'tickers': tickers[:n],
        'expected_returns': er[:n],
        'covariance_matrix': cov[:n,:n],
        'adv': adv[:n]
    }

def _find_issues(covariance_matrix):
    from optimiser_validation_utils import _is_PSD
    #eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    problematic_indices = np.where(eigvals < 1e-10)[0]