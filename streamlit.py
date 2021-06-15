from pandas.core.indexes import multi
import streamlit as st

import numpy as np
import pandas as pd
import pickle
import sklearn

filename = "xgb_reg_H+A_2.pkl"
model = pickle.load(open(filename, "rb"))
X_test = pd.read_csv("X_test.csv", sep=",")
X_test.set_index("Unnamed: 0", inplace=True)
st.text(model.predict(X_test[:1]))
numerical_features= ['Zipcode', 'Surface of the house', 'Terrace Area', 'Garden Area', 'Surface of the land', 'Number of facades', 'Number of rooms']
binary_features = ['Fully equipped kitchen', 'Furnished', 'Open fire',
                       'Swimming pool']
multi_features = ['State of the building', 'Province']
numerical_values = dict()

for f in numerical_features:
    numerical_values[f]=st.number_input(f)

for f in binary_features:
    st.selectbox(f, [ "No", "Yes"])

for v in numerical_values:
    numerical_values[v]
        

