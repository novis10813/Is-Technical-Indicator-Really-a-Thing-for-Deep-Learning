# Is-Technical-Indicator-Really-a-Thing-for-Deep-Learning
This is the report about testing technical indicators whether work or not in deep learning.

## Dataset
I use the crypto 5 min price as the data.

## Model
There will be two models, one is an MLP model and the other one is a model with CNN extractor.
They will use 24 data point (two hours) to predict the close price after one hour, and use the price to determine the buy & sell action.

# MLP model
4 fully-connected layers
64 -> 128 -> 64 -> Output size
