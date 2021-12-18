from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import tensorflow as tf

class DataLabeling:
    """
    This class will label the data with `up`, `down`, `flat` based on the dynamic threshold of close price.
    Threshold:
        if next period close price >= current close price * (1 + alpha * Volatility of last hour), tag it 'up' label
        elif next period close price <= current close price * (1 - alpha * Volatility of last hour), tag it 'down' label
        else tag it with 'flat'
    Base on the three categories, we can detect the trend of the price movement.
    When the tags change from 'down' to 'up' or 'flat' to 'up', it will enter a long trade,
    and the tags change from 'up' to 'down' or 'flat' to 'down', it will enter a short trade,
    otherwise, it will do nothing with a tag of 'Hold'.
    
    Also, it will automatically add features of technical indicators for you.
    The TIs are based on TA-lib
    """
    def __init__(self, data, window_size, alpha=0.55):
        # initialize data and parameters
        self.data = data.loc[:, ['Timestamp' ,'Open', 'High', 'Low', 'Close', 'Volume']]
        self.data = self.data.set_index('Timestamp')
        self.data.index = pd.to_datetime(self.data.index)
        
        self.__alpha = alpha
        self.__window_size = window_size
    
    def __make_label(self, data):
        # Setup a Threshold for Buy, Sell, Hold Label
        data['STD'] = data.Close.rolling(self.__window_size).std()
        data['Next_Close'] = data.Close.shift(-self.__window_size)
        data = data.fillna(0)
        data = data.assign(Trend=data.apply(self.__func, axis=1))
        # data['Trend'] = np.where(data.Next_Close >= data.Close*(1+self.__alpha*data.STD), 1,
        #                               np.where(data.Next_Close <= data.Close*(1-self.__alpha*data.STD), 2, 0))
        data = data.dropna().drop(['Next_Close','STD'], axis=1)
        
        # Normalized the data
        scaler = MinMaxScaler()
        data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
        return data
    
    def __func(self, df):
        if df.Next_Close >= df.Close*(1+self.__alpha*df.STD):
            return 2
        elif df.Next_Close <= df.Close*(1-self.__alpha*df.STD):
            return 0
        else:
            return 1
    
    def __make_TI(self, data):
        data['Chaikin'] = ta.AD(data.High, data.Low, data.Close, data.Volume)
        data['Trange'] = ta.TRANGE(data.High, data.Low, data.Close)
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
        return self.__make_label(self.data)
    
    @property
    def TI_data(self):
        return self.__make_label(self.__make_TI(self.data))

def train_val_test_split(data, train_size=142416, val_size=192, test_size=96, interval=96, rolling=0):
    """
    Split the data into train, val, test datasets with a number of 142416, 192, 96
    and with an interval of 96
    """
    total_size = train_size+val_size+test_size+2*interval+rolling
    if total_size > len(data.iloc[rolling:]):
        print('Out of range, please set rolling smaller')
        exit()
    
    train_df = data.iloc[rolling:train_size]
    val_start = rolling+train_size+interval
    test_start = rolling+train_size+2*interval+val_size
    val_df = data.iloc[val_start:val_start+val_size]
    test_df = data.iloc[test_start:test_start+test_size]

    return train_df, val_df, test_df

class DataPreprocess:
    """
    This class will convert time series data into tf.dataset windowed data
    source: https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing
    """
    def __init__(self, train_df, val_df, test_df,
                 window_size, label_size, label_columns=None, batch_size=12, shift=1):
        
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
        
        self.batch_size = batch_size
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :-1] # exclude Label column
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
            batch_size=self.batch_size)
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

import os
def create_model_checkpoint(model_name, save_path='model_experiments'):
    """
    Create a Model callback to store the best performance model based on val_loss.
    Stores model with the filepath:
        "model_experiments/model_name"
    """
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              verbose=0, # only output text when model is saved
                                              monitor='val_loss',
                                              save_best_only=True)