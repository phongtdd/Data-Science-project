import streamlit as st
import pandas as pd
import json
from UI.Model_Rerun.xgboost.xgboost_model import predict as xg_boost_predict
from UI.Model_Rerun.fcnn.fcnn_model import fcnn_predict

# Function to load data from a JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    df = df.astype({
        'brand': "string",
        'color': "string",
        'size': "string",
        'price': "string",
        'department': "string",
        'origin': "string",
        'rating': "string",
        'star': "string",
        'url': "string"
    })
    return df.reset_index(drop=True)

# Load the data
df = pd.read_json(r"final_data.json")

st.title("Model Selection and Prediction")

# Enter row number
st.header("Select Data")
selected_row = st.number_input(
    "Enter a Row Number", 
    min_value=0, 
    max_value=len(df) - 1, 
    step=1, 
    value=0
)

st.write("Selected Row Infomation:")
row_data = df.iloc[[selected_row]]
if not row_data.empty:
    st.dataframe(row_data, width=2000, height=50)
else:
    st.write("No data available for the selected row.")

# Select Model
st.header("Select Model")
model = st.selectbox("Select Model", ["FCNN" ,"XGBoost", "Decision Tree", "Random Forest", "MLP"])


if st.button("Predict"):
    if model == "Decision Tree":
        result = "Prediction from Decision Tree"
    elif model == "Random Forest":
        result = "Prediction from Random Forest"
    elif model == "XGBoost":
            predicted_price = xg_boost_predict(selected_row)
            
    elif model == "FCNN":
            predicted_price = fcnn_predict(selected_row)
    elif model == "MLP":
        result = "Prediction from Model MLP"
        
    st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True); st.write(predicted_price)
    
    

