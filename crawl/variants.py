import re
import json
import random
import asyncio
import httpx
from search import SEARCH_QUERY
import os


REMAINING_ASIN_FILE = f"data/{SEARCH_QUERY}/remaining_{SEARCH_QUERY}_asin.json"
VARIANT_PRODUCT_ASIN_FILE = f"data/{SEARCH_QUERY}/variant_{SEARCH_QUERY}_asin.json"


BASE_HEADERS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 11; SM-A125U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5672.63 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.110 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5914.96 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6034.198 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (iPad; CPU OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 11.0; ARM64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6065.170 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
]

# Load ASINs from JSON file

# Semaphore for limiting concurrent requests
semaphore = asyncio.Semaphore(12)  # Limit to 8 concurrent requests


async def fetch_product_data(asin: str, session: httpx.AsyncClient):
    """Fetch and parse product data."""
    
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
    except Exception as e:
        print(f"Error processing ASIN {asin}: {e}")

async def process_asins(asin_list):
    """Process each ASIN concurrently."""
    async with httpx.AsyncClient() as session:
        print(f'Remain: {len(list(set(asin_list)))}')
        # Loop though each ASIN and fetch the data
        tasks = [fetch_product_data(asin, session) for asin in list(set(asin_list))]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        with open(VARIANT_PRODUCT_ASIN_FILE, "a") as output_file:
            json.dump(cleaned_entry, output_file, indent=4)

        for asin in list(set(asin_list)):
            if asin not in cleaned_entry:
                fail_asin.append(asin)
        with open(REMAINING_ASIN_FILE, "w") as output_file:
            json.dump(list(set(fail_asin)), output_file, indent=4)

condition = True
while condition:
    cleaned_entry ={}
    fail_asin = []
    with open(REMAINING_ASIN_FILE, 'r') as file:
        asin_list = json.load(file)
        # print(asin_list)
        # print(len(list(set(asin_list))))
    
    if len(list(set(asin_list)))<10:
        condition = False
    asyncio.run(process_asins(asin_list))
    print(f"Merged data saved to {VARIANT_PRODUCT_ASIN_FILE}")
