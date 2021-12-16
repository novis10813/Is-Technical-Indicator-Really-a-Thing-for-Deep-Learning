# This program is used to backtest the strategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Metric
import tensorflow as tf
plt.style.use('ggplot')

from backtesting import Strategy, Backtest

class Backtest:
    def __init__(self, df=None, comission_fee=0.00075, initial_balance=0.001):
        # backtesting for every two hours
        self.df = self.__backtest_preprocess(df[23::24])
        self.comission_fee = comission_fee
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.crypto_held = 0
        
        # Draw return
        self.data = df
    
    def test(self):
        for i in range(len(self.df)):
            if self.df['Signal'][i] == 'Hold':
                pass
            elif self.df['Signal'][i] == 'Buy' and self.balance > self.initial_balance/100:
                self.crypto_bought = (self.balance/self.df['Close'][i]) * (1 - self.comission_fee)
                self.balance -= self.crypto_bought * self.df['Close'][i]
                self.crypto_held += self.crypto_bought
            
            elif self.df['Signal'][i] == 'Sell' and self.crypto_held > 0:
                self.crypto_sold = self.crypto_held
                self.balance += self.crypto_sold * self.df['Close'][i] * (1 - self.comission_fee)
                self.crypto_held -= self.crypto_sold
        print(f'balance: {self.balance} | crypto: {self.crypto_held}')
        return self.balance
    
    def __backtest_preprocess(self, data):
        data['Processed_Trend'] = data.Trend.replace(to_replace=1, method='bfill')
        data['Previous_Trend'] = data.Processed_Trend.shift()
        data = data.assign(Signal=data.apply(self.__func, axis=1))
        return data

    def __func(self, data):
        if data['Processed_Trend'] > data['Previous_Trend']:
            return 'Buy'
        elif data['Processed_Trend'] < data['Previous_Trend']:
            return 'Sell'
        else:
            return 'Hold'
    

class WeightedFScore(Metric):
    """
    calculate three types of state
    1. Actual price movement is up/down but the model predicts down/up
    2. Actual price movement is up/down but the model predicts flat
    3. Actual price movement is flat but the model predicts up/down
    The first situation will br given a weight of 0.5,
    the second and the third wil be given 0.125
    """
    def __init__(self, *args, **kwargs):
        super(WeightedFScore, self).__init__(name='WeightedFScore')
        self.beta_1 = 0.5
        self.beta_2 = 0.125
        self.beta_3 = 0.125
    
    def update_state(self, y_true, y_pred):
        array = tf.math.confusion_matrix(y_true, y_pred).numpy()
        true_positive = 0.125*0.125*array[0, 0] + array[1, 1] + array[2, 2]
        type_1_error = array[1, 2] + array[2, 1]
        type_2_error = array[1:, 0].sum()
        type_3_error = array[0, 1:].sum()
        
        self.F.assign((1+self.beta_1**2 + self.beta_2**2)*true_positive / ((1+self.beta_1**2 + self.beta_2**2)*true_positive + type_1_error + self.beta_1**2 * type_2_error + self.beta_2**2 * type_3_error))
        
    def result(self):
        return self.F
        
    def reset_states(self):
        self.F.assign(0)