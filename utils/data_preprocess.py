import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import tensorflow as tf

class DataLabeling:
    """
    This class will label the data with `Buy` `Sell` or `Hold` based on the dynamic threshold of log return.
    Threshold:
        if next close price >= current close price * (1 + alpha * Volatility of last hour), tag it 'up' label
        elif next close price <= current close price * (1 - alpha * Volatility of last hour), tag it 'down' label
        else tag it with 'flat'
    Base on the three categories, we can detect the trend of the price movement.
    When the tags change from 'down' to 'up' or 'flat' to 'up', it will enter a long trade with a tag of 'Buy',
    and the tags change from 'up' to 'down' or 'flat' to 'down', it will enter a short trade with a tag of 'Sell',
    otherwise, it will do nothing with a tag of 'Hold'.
    
    Also, it will automatically add features of technical indicators for you.
    The TIs are based on TA-lib
    """
    def __init__(self, data, volatility_period, alpha=0.55):
        # initialize data and parameters
        self.data = data.set_index('Timestamp').loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
        self.alpha = alpha
        self.volatility_period = volatility_period
    
    def make_label(self, data):
        # Setup a Threshold for Buy, Sell, Hold Label
        data['Trend'] = np.where(data.Close >= data.Close.shift(1)*(1+self.alpha*data.Close.rolling(self.volatility_period).std()), 1,
                                      np.where(data.Close <= data.Close.shift(1)*(1-self.alpha*self.data.Close.rolling(self.volatility_period).std()), -1, 0))
        data = data.dropna()
        data['Label'] = np.where(data.Trend < data.Trend.shift(-1), 'Buy',
                                 np.where(data.Trend > data.Trend.shift(-1), 'Sell', 'Hold'))
        
        data = data.dropna().drop(['Trend'], axis=1)
        return data
    
    def make_TI(self, data):
        data['Chaikin'] = ta.AD(data.High, data.Low, data.Close, data.Volume)
        data['Trange'] = ta.TRANGE(data.High, data.Low, data.Close)
        data['Hammer'] = ta.CDLHAMMER(data.Open, data.High, data.Low, data.Close)
        data['ShootingStar'] = ta.CDLSHOOTINGSTAR(data.Open, data.High, data.Low, data.Close)
        for i in [5, 7, 14, 30]:
            data[f'RSI_{i}'] = ta.RSI(data.Close, timeperiod=i)
            data[f"DX_{i}"] = ta.DX(data.High, data.Low, data.Close, timeperiod=i)
            data[f"ADX_{i}"] = ta.ADX(data.High, data.Low, data.Close, timeperiod=i)
            data[f"ADXR_{i}"] = ta.ADXR(data.High, data.Low, data.Close, timeperiod=i)
            data[f'EMA_{i}'] = ta.EMA(data.Close, timeperiod=i)
            data[f'BBand_upper_{i}'], _, data[f'BBand_lower_{i}'] = ta.BBANDS(data.Close, timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)
            data[f'DEMA_{i}'] = ta.DEMA(data.Close, timeperiod=i)
            data[f'NATR_{i}'] = ta.NATR(data.High, data.Low, data.Close, timeperiod=i)
            
        return data
    
    @property
    def labelled_data(self):
        return self.make_label(self.data)
    
    @property
    def TI_data(self):
        return self.make_label(self.make_TI(self.data))
        
def data_split(data, kfolds):
    """
    Create a function to split the data into subsets
    
    Parameters:
        data: a pandas dataframe
        kfolds: how many datasets you want
    Returns:
        few dataset
    """
    return np.array_split(data, kfolds)

def train_val_test_split(data, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Split the data into train, val, test datasets
    Warnings: The sum of the ratio should be 1 or there might be data leakage problem.
    """
    n = len(data)
    train_df = data[:int(n*train_size)]
    val_df = data[int(n*train_size):int(n*(train_size+val_size))]
    test_df = data[int(n*(1-test_size)):]
    
    return train_df, val_df, test_df

class DataPreprocess:
    """
    This class will convert time series data into tf.dataset windowed data
    source: https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing
    """
    def __init__(self, train_df, val_df, test_df,
                 window_size, label_size, label_columns=None, shift=1):
        
        # raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # label column
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        
        self.label_columns_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        # window parameters
        self.window_size = window_size
        self.label_size = label_size
        self.shift = shift
        
        self.total_window_size = window_size + shift
        
        self.input_slice = slice(0, window_size)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_size
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.label_columns_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.window_size, None])
        labels.set_shape([None, self.label_size, None])
        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=128)
        ds = ds.map(self.split_window)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)

def buy_sell_threshold(data, alpha):
    data['Log_return'] = np.log(data.Close).diff()
    
    for i in range(len(data)):
        if data.Log_return[i+1] >= data.Log_return[i](1+alpha*(data.Close.rolling(10).std())):
            data['State'] = 'Buy'
        elif data.Log_return[i+1] <= data.Log_return[i](1 - alpha*data.Close.rolling(10).std()):
            data['State'] = 'Sell'
        else:
            data['State'] = 'Hold'
    
    return data