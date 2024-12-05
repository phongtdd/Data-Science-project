import json
from variants import VARIANT_PRODUCT_ASIN_FILE,SEARCH_QUERY

FILTERED_PRODUCT_ASIN_FILE = f"data/{SEARCH_QUERY}/filtered_{SEARCH_QUERY}_asin.json"

def filter_asins(data, color_filters, size_filters):

    filtered = {}

    for asin, attributes in data.items():
        if len(attributes)!=2:
            continue
        # print(attributes)
        color, size = attributes  # Unpack the color and size
         # Check if any color filter is a substring of the color attribute
        color1_match = len(color.split())==1
        color2_match = len(size.split())==1

        # Check if any size filter is a substring of the size attribute
        size1_match = size.lower() in size_filters
        size2_match = color.lower() in size_filters

        # If both color and size match, add the ASIN to the filtered list
        if color1_match and size1_match:
            filtered[asin]=attributes
        elif color2_match and size2_match:
            filtered[asin]=attributes
        
    return filtered


color_filters = ["Black", "White", "Gray", "Beige", "Red", "Blue", "Green", "Brown", "Pink", "Purple", "Yellow"]
size_filters = ["small", "medium", "large", "x-large", "xx-large"]

with open(VARIANT_PRODUCT_ASIN_FILE,'r') as file:
    data = json.load(file)
    print(len(data))


filtered_asins = filter_asins(data, color_filters, size_filters)

with open(FILTERED_PRODUCT_ASIN_FILE,'w') as file:
    json.dump(filtered_asins,file,indent=4)
    