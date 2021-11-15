import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import tensorflow as tf

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

def buy_sell_threshold():
    pass