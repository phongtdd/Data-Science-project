import re
import json
import random
import asyncio
import httpx
from search import SEARCH_QUERY
from extract_asin import REMAINING_ASIN_FILE

VARIANT_PRODUCT_ASIN_FILE = f"variant_{SEARCH_QUERY}_asin.json"


# Headers to mimic a browser
BASE_HEADERS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:112.0) Gecko/20100101 Firefox/112.0"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69"},
]

# Load ASINs from JSON file
with open(REMAINING_ASIN_FILE, 'r') as file:
    asin_list = json.load(file)

# Semaphore for limiting concurrent requests
semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent requests

cleaned_entry ={}
fail_asin = []

async def fetch_product_data(asin: str, session: httpx.AsyncClient):
    """Fetch and parse product data."""
    try:
        # Rotate headers
        headers = random.choice(BASE_HEADERS)

        # Fetch product HTML
        product_html = await session.get(f"https://www.amazon.com/dp/{asin}", headers=headers)

        # Extract JSON-like data using regex
        variant_data = re.findall(r'dimensionValuesDisplayData"\s*:\s*({.+?}),\n', product_html.text)

        # Parse and write each entry to the output file
        for entry in variant_data:
            parsed_entry = json.loads(entry)
            
            # Extract only the key-value pairs and save
            for key,values in parsed_entry.items():
                cleaned_entry[key] = values
    except Exception as e:
        print(f"Error processing ASIN {asin}: {e}")
        fail_asin.append(asin)

async def process_asins():
    """Process each ASIN concurrently."""
    async with httpx.AsyncClient() as session:
        # Loop through each ASIN and fetch the data
        tasks = [fetch_product_data(asin, session) for asin in asin_list]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        with open(VARIANT_PRODUCT_ASIN_FILE, "a") as output_file:
            json.dump(cleaned_entry, output_file, indent=4)
        with open(REMAINING_ASIN_FILE, "w") as output_file:
            json.dump(fail_asin, output_file, indent=4)
            output_file.write("\n")  # Write a newline for separation between records

if __name__ == "__main__":
    # Run the asynchronous process
    asyncio.run(process_asins())
    print(f"Merged data saved to {VARIANT_PRODUCT_ASIN_FILE}")
