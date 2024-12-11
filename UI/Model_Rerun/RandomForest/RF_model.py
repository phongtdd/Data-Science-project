import pandas as pd
import joblib
import streamlit
def predict(X):
    model = joblib.load(r"UI\Model_Rerun\DecisionTree\decision_tree_price_prediction_model.pkl")
    predictions = model.predict(X)
    return predictions[0]