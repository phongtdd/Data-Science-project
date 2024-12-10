import pandas as pd
import numpy as np
import joblib

def predict(index):
    df = pd.read_pickle(r"UI\Model_Rerun\xgboost\xg_boost_processed_data.pkl")
    row = df.iloc[index]
    row_df = pd.DataFrame([row])
    row_df.drop(['log_price'], axis=1, inplace=True)
    model = joblib.load(r'UI\Model_Rerun\xgboost\xgboost_model.pkl')
    predictions = model.predict(row_df)
    predicted_prices = np.exp(predictions).item() - 1
    return predicted_prices