import joblib
import streamlit as st

# Function to encode brand
def encode_brand(brand: str):
    loaded_encoder = joblib.load(r'label_encode\brand_encoder.pkl')
    if brand in loaded_encoder.classes_:  # Kiểm tra xem nhãn có trong tập các nhãn đã học hay không
        encoded_label = loaded_encoder.transform([brand])
        return encoded_label[0]
    else:
        return None

# Function to encode size
def encode_size(size):
    size_map = {'Small':0, 'Medium':1, 'Large':2, "X-Large":3, "XX-Large":4}
    if size in size_map:
        return size_map[size]
    else:
        return None

# Function to encode department
def encode_department(depart):
    department_map = {'Women':1, 'Men':2}
    if depart in department_map:
        return department_map[depart]
    else:
        return None

# Function to encode color
def encode_color(col):
    loaded_encoder = joblib.load(r'label_encode\color_encoder.pkl')
    if col in loaded_encoder.classes_:  # Check if the color is in the encoder's learned classes
        encoded_label = loaded_encoder.transform([col])
        return encoded_label[0]
    else:
        return None

# Function to encode origin
def encode_origin(ori):
    ori_map = {'Imported':0, 'Made in USA':1}
    if ori in ori_map:
        return ori_map[ori]
    else:
        return None
