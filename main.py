# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:13:10 2021

@author: jerem
"""

import pandas as pd
import xgboost as xgb

data = pd.read_csv("Data4.csv", sep=",")
print(data.columns)
data.set_index(data["Unnamed: 0"])
print(data.info())

# read in data
dtrain = xgb.DMatrix(data.iloc[:50000])
dtest = xgb.DMatrix(data.iloc[50000:])
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
