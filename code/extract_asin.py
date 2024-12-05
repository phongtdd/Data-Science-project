import json
from search import SEARCH_QUERY, UNIQUE_PRODUCT_SAVE_FILE

UNIQUE_PRODUCT_SAVE_FILE = f"data/{SEARCH_QUERY}/amazon_unique_{SEARCH_QUERY}.json"
REMAINING_ASIN_FILE = f"data/{SEARCH_QUERY}/remaining_{SEARCH_QUERY}_asin.json"

# Load the JSON file
with open(UNIQUE_PRODUCT_SAVE_FILE, "r") as file:
    data = json.load(file)

# Extract all ASIN values
asin_values = [item["asin"] for item in data]

with open(REMAINING_ASIN_FILE,'w') as f:
    json.dump(asin_values,f,indent=4)
    
print("Extract ASIN File Path:", REMAINING_ASIN_FILE)
