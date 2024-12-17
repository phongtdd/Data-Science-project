import pandas as pd
import numpy as np
import joblib
import streamlit as st

def predict(X):
    model = joblib.load(r'UI\Model_Rerun\xgboost\xgboost_model.pkl')
    prediction = model.predict(X)
    return float(prediction[0])