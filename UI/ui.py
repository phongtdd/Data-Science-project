import streamlit as st
import pandas as pd
import json
from UI.Model_Rerun.xgboost.xgboost_model import predict as xg_boost_predict
from UI.Model_Rerun.fcnn.fcnn_model import predict as fcnn_predict
from UI.Model_Rerun.DecisionTree.DT_model import predict as dt_predict
from UI.Model_Rerun.RandomForest.RF_model import predict as rf_predict
from UI.Model_Rerun.MLP.mlp_model import predict as mlp_predict

def make_prediction(df):
    expected_features = ['brand', 'color', 'size', 'department', 'origin', 'rating', 'star', 'Polyester',
 'Spandex', 'Nylon', 'Cotton', 'Rayon', 'Acrylic', 'Modal', 'Wool', 'Lyocell', 
 'Leather', 'Linen', 'Silk', 'Machine Wash', 'Hand Wash', 'Not Bleach',
 'Tumble Dry', 'Pull on', 'Tie', 'Zipper', 'Button', 'No closure', 'Elastic',
 'Lace Up', 'Drawstring']
    df = df[expected_features]
    
    # columns_to_drop = ['name', 'asin', 'url', 'price']
    # df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
    
    # Select Model
    model = st.selectbox("Select Model", ["FCNN" ,"XGBoost", "Decision Tree", "Random Forest", "MLP"],
                         index=None, placeholder="Select a Model")

    if st.button("Predict"):
        predicted_price = 0
        if model == "Decision Tree":
            predicted_price = dt_predict(df)
            st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True)
            
        elif model == "Random Forest":
            predicted_price = rf_predict(df)
            st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True)

        elif model == "XGBoost":
            predicted_price = xg_boost_predict(df)
            st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True)

        elif model == "FCNN":
            predicted_price = fcnn_predict(df)
            st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True)

        elif model == "MLP":
            predicted_price = mlp_predict(df)
            st.markdown("<b>Prediction Result:</b>", unsafe_allow_html=True)

        st.write(round(predicted_price, 2))
        
        
def get_manual_input():
    columns = ['brand', 'size', 'department', 'color', 'origin', 'star']
    
    # Tạo dictionary để lưu dữ liệu nhập vào
    input_data = {}
    for col in columns:
        value = input(f"Enter value for '{col}': ")
        input_data[col] = value
    
    # Chuyển dữ liệu thành DataFrame
    df_manual = pd.DataFrame([input_data])
    return df_manual


# Get data
original_data = pd.read_json(r"UI\final_data.json")

df = pd.read_json(r"UI\preprocess_data.json")

def hide_input():
    st.session_state.show_input = False

if "show_input" not in st.session_state:
    st.session_state.show_input = True

st.title("Model Selection and Prediction")

model = st.selectbox("Select Data Entry Method", 
                     ["Choose a product in database", "Enter the data manually"], 
                     index=None, placeholder="Select an option")

if model == "Choose a product in database":
    # Enter row number
    st.markdown("<b style='font-size:24px;'>Select Data:</b>", unsafe_allow_html=True)
    selected_row = st.number_input(
        "Enter a Row Number", 
        min_value=0, 
        max_value=len(df) - 1, 
        step=1, 
        value=0
    )

    st.write("Selected Row Infomation:")
    # Lấy hàng được chọn
    row_data_4display = original_data.iloc[[selected_row]]
    row_data = df.iloc[[selected_row]]
    
    columns_to_display = ['name','brand', 'size', 'department', 'color', 'origin', 'star', 'price', 'url']
    filtered_row_data = row_data_4display[columns_to_display]
    if not filtered_row_data.empty:
        st.dataframe(filtered_row_data, width=2000, height=50)
        make_prediction(row_data)
    else:
        st.write("No data available for the selected row.")
    

elif model == "Enter the data manually":
    st.markdown("<b style='font-size:24px;'>Enter Data for Prediction:</b>", unsafe_allow_html=True)

    columns = ['brand', 'color', 'size', 'department', 'origin']
    num_columns = ['rating', 'star']
    material = ['Polyester', 'Spandex', 'Nylon', 'Cotton', 'Rayon', 'Acrylic', 'Modal', 'Wool', 'Lyocell', 'Leather', 'Linen', 'Silk']
    care_instruction = ['Machine Wash', 'Hand Wash', 'Not Bleach', 'Tumble Dry']
    others = ['Pull on', 'Tie', 'Zipper', 'Button', 'No closure', 'Elastic', 'Lace Up', 'Drawstring']
    
    input_data = {}
    
    for col in num_columns:
        input_data[col] = 0
    for mat in material:
        input_data[mat] = 0
    for care in care_instruction:
        input_data[care] = 0
    for other in others:
        input_data[other] = 0
        
    for col in columns:
        # Have to double check value if available
        value = st.text_input(f"Enter {col}:")
        input_data[col] = value
        
    for col in num_columns:
        value = st.text_input(f"Enter {col}:")
        input_data[col] = value
              
    selected_care_instructions = st.multiselect("Select care instructions:", care_instruction)
    for instruction in selected_care_instructions:
        input_data[instruction] = 1

    selected_other_features = st.multiselect("Select other features:", others)
    for feature in selected_other_features:
        input_data[feature] = 1
         
    if "material_inputs" not in st.session_state:
        st.session_state.material_inputs = []

    def display_material_inputs():
        for i, (material, content) in enumerate(st.session_state.material_inputs):
            cols = st.columns(2)
            with cols[0]:
                st.selectbox(
                    f"Material {i + 1}",
                    ['Polyester', 'Spandex', 'Nylon', 'Cotton', 'Rayon', 'Acrylic', 'Modal', 'Wool', 'Lyocell', 'Leather', 'Linen', 'Silk'],
                    index=0 if material is None else ['Polyester', 'Spandex', 'Nylon', 'Cotton', 'Rayon', 'Acrylic', 'Modal', 'Wool', 'Lyocell', 'Leather', 'Linen', 'Silk'].index(material),
                    key=f"material_{i}",
                )
            with cols[1]:
                st.number_input(
                    f"Content (%)",
                    min_value=0,
                    max_value=100,
                    value=content if content is not None else 0,
                    key=f"content_{i}",
                )

    def add_material_button():
        if st.button("Add Material"):
            st.session_state.material_inputs.append((None, None))
    
    add_material_button()
    display_material_inputs()
    if st.button("Submit", on_click=hide_input):
        for i in range(len(st.session_state.material_inputs)):
            material = st.session_state.get(f"material_{i}")
            content = st.session_state.get(f"content_{i}")
            if material is not None and content is not None:
                input_data[material] = round(int(content) / 100, 2)

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame([input_data])
    df = df.apply(pd.to_numeric, errors='coerce')
    st.write(df)
    make_prediction(df)

    
    
    








