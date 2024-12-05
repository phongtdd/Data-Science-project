import re
import json
import random
import asyncio
import httpx
from search import SEARCH_QUERY
from extract_asin import REMAINING_ASIN_FILE

VARIANT_PRODUCT_ASIN_FILE = f"data/{SEARCH_QUERY}/variant_{SEARCH_QUERY}_asin.json"


# Headers to mimic a browser
BASE_HEADERS=[
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_7_9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.126 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G998U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.121 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 12; Pixel 6 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.171 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; ARM64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.125 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.5304.87 Safari/537.36"}
]


# Load ASINs from JSON file

# Semaphore for limiting concurrent requests
semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent requests

all_entry ={}
fail_asin = []
async def fetch_product_data(asin: str, session: httpx.AsyncClient):
    """Fetch and parse product data."""
    cleaned_entry ={}
    try:
        # Rotate headers
        headers = random.choice(BASE_HEADERS)

        # Fetch product HTML
        product_html = await session.get(f"https://www.amazon.com/dp/{asin}", headers=headers)

        # Extract JSON-like data using regex
        variant_data = re.findall(r'dimensionValuesDisplayData"\s*:\s*({.+?}),\n', product_html.text)
        if len(variant_data)!=0:
        # Parse and write each entry to the output file
            for entry in variant_data:
                parsed_entry = json.loads(entry)
                
                # Extract only the key-value pairs and save
                for key,values in parsed_entry.items():
                    cleaned_entry[key] = values
            all_entry[asin] = cleaned_entry
    except Exception as e:
        print(f"Error processing ASIN {asin}: {e}")

async def process_asins(asin_list):
    """Process each ASIN concurrently."""
    async with httpx.AsyncClient() as session:
        print(len(list(set(asin_list))))
        # Loop through each ASIN and fetch the data
        tasks = [fetch_product_data(asin, session) for asin in list(set(asin_list))]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        with open(VARIANT_PRODUCT_ASIN_FILE, "a") as output_file:
            json.dump(all_entry, output_file, indent=4)

        for asin in list(set(asin_list)):
            if asin not in all_entry:
                fail_asin.append(asin)
        with open(REMAINING_ASIN_FILE, "w") as output_file:
            json.dump(list(set(fail_asin)), output_file, indent=4)

if __name__ == "__main__":
    # Run the asynchronous process
    print(REMAINING_ASIN_FILE)
    with open(REMAINING_ASIN_FILE, 'r') as file:
        asin_list = json.load(file)

    asyncio.run(process_asins(asin_list))
    print(f"Merged data saved to {VARIANT_PRODUCT_ASIN_FILE}")
