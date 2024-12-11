import joblib

# Function to encode brand
def encode_brand(brand:str):
    loaded_encoder = joblib.load('label_encode/brand_encoder.pkl')
    encoded_label =  loaded_encoder.transform([brand])
    return encoded_label[0]

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
    loaded_encoder = joblib.load('label_encode/color_encoder.pkl')
    encoded_label =  loaded_encoder.transform([col])
    return encoded_label[0]

# Encode to encode origin
def encode_origin(ori):
    ori_map = {'Imported':0,"Made in USA":1}
    return ori_map(ori)