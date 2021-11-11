import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocess:
    """
    This class helps you to view and clean the data download from Binance
    """
    def __init__(self, path:str):
        data = pd.read_csv(path)
        data = data.set_index('Timestamp')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :]
    
    def get_ohlcv(self):
        return self.data.iloc[:, :5]