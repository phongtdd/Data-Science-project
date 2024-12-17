import pandas as pd
import joblib

def predict(X): 
    model = joblib.load(r"UI\Model_Rerun\DecisionTree\decision_tree_price_prediction_model.pkl")
    prediction = model.predict(X)
    predicted_value = prediction.item()  
    return predicted_value