import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta

class DataPreprocess:
    """
    This class helps you to view and clean the data download from Binance
    """
    def __init__(self, path:str):
        data = pd.read_csv(path)
        self.data = data.set_index('Timestamp')
    
    def get_ohlcv(self):
        return self.data.iloc[:, :5]
    
# Create a function to split the data into subsets
def data_split(data, kfolds):
    """
    Parameters:
        data: a pandas dataframe
        kfolds: how many datasets you want
    Returns:
        few dataset
    """
    return np.array_split(data, kfolds)

# Create a function to make a windowed data
def make_windows(data, window_size):
    """
    Parameters:
        data: a pandas dataframe, ['Close'] is required.
        window_size: size of the windows you use to predict.
    Returns:
        a pandas dataframe
    """
    for i in range(window_size):
        data[f'Close{window_size-i}'] = data['Close'].shift(periods=i+1)
    
    X = data.dropna().drop('Close', axis=1).astype(np.float32)
    y = data.dropna()['Close'].astype(np.float32)
    
    return X, y

def buy_sell_threshold():
    pass