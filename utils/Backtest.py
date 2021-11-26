# This program is used to backtest the strategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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