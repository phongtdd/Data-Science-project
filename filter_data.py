import json
from variants import VARIANT_PRODUCT_ASIN_FILE,SEARCH_QUERY

FILTERED_PRODUCT_ASIN_FILE = f"filtered_{SEARCH_QUERY}_asin.json"

def filter_asins(data, color_filters, size_filters):

    filtered_asins = []

    for asin, attributes in data.items():
        if len(attributes)>2:
            continue
        # print(attributes)
        color, size = attributes  # Unpack the color and size
         # Check if any color filter is a substring of the color attribute
        color1_match = any(filter_color.lower() in color.lower() for filter_color in color_filters)
        color2_match = any(filter_color.lower() in size.lower() for filter_color in color_filters)

        # Check if any size filter is a substring of the size attribute
        size1_match = any(filter_size.lower() in size.lower() for filter_size in size_filters)
        size2_match = any(filter_size.lower() in color.lower() for filter_size in size_filters)

        # If both color and size match, add the ASIN to the filtered list
        if color1_match and size1_match:
            filtered_asins.append(asin)
        elif color2_match and size2_match:
            filtered_asins.append(asin)
        
    return filtered_asins


color_filters = ["Black", "White", "Gray", "Beige", "Red", "Blue", "Green", "Brown", "Pink", "Purple", "Yellow"]
size_filters = ["Small", "Medium", "Large", "X-Large", "XX-Large"]

with open(VARIANT_PRODUCT_ASIN_FILE,'r') as file:
    data = json.load(file)
    print(len(data))


filtered_asins = filter_asins(data, color_filters, size_filters)

with open(FILTERED_PRODUCT_ASIN_FILE,'w') as file:
    json.dump(filtered_asins,file,indent=4)
    