# This program is used to backtest the strategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Metric
import tensorflow as tf
plt.style.use('ggplot')

class Backtesting:
    def __init__(self, data, bitcoin_amount, asset=0, percent=1):
        self.data = data # labeled data
        self.bitcoin_amount = bitcoin_amount #
        self.asset = asset # target asset you hold
        self.percent = percent # percentage of asset you buy and sell
        
    def Buy_Hold_Sell(self):
        
        for i in range(len(self.data)):
            if self.data['Label'][i] == 0: # Hold
                pass
            elif self.data['Label'][i] == 1: # Buy
                self.asset += (self.percent*self.bitcoin_amount)/self.data['Close']
                self.bitcoin_amount -= self.percent*self.bitcoin_amount
            else:
                self.asset -= self.percent*self.asset
                self.bitcoin_amount += self.percent*self.asset*self.data['CLose']
                pass
    
    # def plot_history(self):
    #     plt.figure(figsize=(15, 12))
    #     plt.plot(self.data.Close)
    #     plt.plot(self.data.Label == 1, '^g')
    #     plt.plot(self.data.Label == 2, 'vr')
    #     plt.legend('upperleft')
    #     plt.xlabel('Time')
    #     plt.ylabel('Price')
    #     plt.show()
    

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