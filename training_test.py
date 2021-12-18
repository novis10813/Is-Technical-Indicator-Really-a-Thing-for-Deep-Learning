import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_preprocess import * # data_split, DataLabeling, train_val_test_split
from models import *

WINDOW_SIZE = 24

df = pd.read_csv('data\ETHBTC-5m-data.csv')
raw_data = DataLabeling(df, WINDOW_SIZE)
train_df, val_df, test_df = train_val_test_split(raw_data.labelled_data)

Data = DataPreprocess(train_df, val_df, test_df, step=24)
X_train, y_train = Data.train
X_val, y_val = Data.val
X_test, y_test = Data.test

model = CDT_1D_model(WINDOW_SIZE, 5)
test_model = model.model
test_model.fit(X_train, y_train, batch_size=12, epochs=20, validation_data=(X_val, y_val), verbose=0)
test_model.evaluate(X_test, y_test)