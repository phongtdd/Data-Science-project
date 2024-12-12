import joblib
import streamlit as st

# Function to encode brand
def encode_brand(brand: str):
    loaded_encoder = joblib.load(r'label_encode\brand_encoder.pkl')
    st.write(loaded_encoder.classes_)
    st.write(brand in loaded_encoder.classes_)
    if brand in loaded_encoder.classes_:  # Kiểm tra xem nhãn có trong tập các nhãn đã học hay không
        encoded_label = loaded_encoder.transform([brand])
        return encoded_label[0]
    else:
        st.warning(f"Brand '{brand}' not recognized, please enter a valid brand name.")
        return -1  # Trả về -1 nếu không có nhãn
    
#Function to encode size
def encode_size(size):
    size_map = {'Small':0, 'Medium':1, 'Large':2, "X-Large":3, "XX-Large":4}
    return size_map[size]

#Function to encode department
def encode_department(depart):
    department_map = {'Women':1,'Men':2}
    return department_map[depart]

# Function to encode color
def encode_color(col):
    loaded_encoder = joblib.load(r'label_encode\color_encoder.pkl')
    encoded_label =  loaded_encoder.transform([col])
    return encoded_label[0]

# Encode to encode origin
def encode_origin(ori):
    ori_map = {'Imported':0,"Made in USA":1}
    return ori_map(ori)