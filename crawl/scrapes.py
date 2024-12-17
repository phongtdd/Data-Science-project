import asyncio
import httpx
import random
import json
from typing_extensions import TypedDict
from parsel import Selector
from typing import List
import time
from search import SEARCH_QUERY
import os

FILTERED_PRODUCT_ASIN_FILE = f"data/{SEARCH_QUERY}/filtered_{SEARCH_QUERY}_asin.json"
FAIL_TO_LOAD_ASIN_FILE = f"data/{SEARCH_QUERY}/fail_{SEARCH_QUERY}_asin.json"
PRODUCT_INFO_FILE = f"data/{SEARCH_QUERY}/{SEARCH_QUERY}_info.json"

if not os.path.exists(FAIL_TO_LOAD_ASIN_FILE):
    
    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(FAIL_TO_LOAD_ASIN_FILE), exist_ok=True)
    
    # Write the default data to the new file
    with open(FAIL_TO_LOAD_ASIN_FILE, "w") as file:
        json.dump('default_data', file, indent=4)

class ProductInfo(TypedDict):
    """type hint for our scraped product result"""
    name: str
    asin: str
    brand: str
    color: str
    size: str
    price: str
    fabric: str
    care: str
    department: str
    origin: str
    closure: str
    first_date: str
    rating: str
    star: str
    url: str
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 12; SM-A716V) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.2045.31",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; ARM64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; Mi 9T Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; Nexus 5X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/115.0.2045.31",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 6 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Linux; Android 8.1; Pixel 2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.5481.77 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-G998U1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/118.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.5678.93 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-N970F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 16_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.5414.119 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; SM-A515F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; ARM64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 9; Nexus 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.5993.88 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_6_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.5414.97 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; SM-G960F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/119.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 12; Pixel 6a) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
]




def get_remain_asin(current_file_path, all_asin_path):
    remain_asin = {}
    with open(current_file_path,'r') as f:
        data = json.load(f)
    current_asin_list = set([item["asin"] for item in data])
    with open(all_asin_path,'r') as f:
        all_asin_data = json.load(f)
        asin_list = list(all_asin_data.keys())
    for asin in asin_list:
        if asin not in current_asin_list:
            remain_asin[asin] = all_asin_data[asin]
    return remain_asin

# Function to extract keys (used later in run)
def extract_keys(obj, parent_key=""):
    keys = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            keys.append(full_key)
            keys.extend(extract_keys(value, full_key))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            keys.extend(extract_keys(item, f"{parent_key}[{index}]"))
    return keys

def parse_product(result, asin, attribute) -> ProductInfo:
    """parse Amazon's product page (e.g. https://www.amazon.com/dp/B07KR2N2GF) for essential product data"""

    sel = Selector(text=result.text)

    # Extract details from "Product Information" table
    department = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Department')]/following-sibling::span/text()").get()
    care = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Care instructions')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if care is None:
        care = "None"
    fabric = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Fabric type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if fabric is None:
        fabric = "None"
    origin = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Origin')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if origin is None:
        origin = "None"    
    closure_types = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Closure type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if closure_types is None:
        closure_types = "None"      
    first_date = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Date First Available')]/following-sibling::span/text()").get()

    box = sel.css("#centerCol")
    
    brand = box.xpath("//*[@id='bylineInfo']/text()").get()
    price = box.css(".a-price.a-text-price.a-size-medium.apexPriceToPay > span.a-offscreen::text").get()
    
    review = sel.xpath("//*[@id='acrCustomerReviewText']/text()").get()
    star = sel.xpath("//*[@id='acrPopover']/span[1]/a/span/text()").get()
    if star is None:
        star = "None"
    
    parsed = {
        "name": sel.css("#productTitle::text").get("").strip(),
        "asin": asin,
        "brand": brand,
        "color": attribute[0],
        "size": attribute[1],
        "price": price,
        "fabric": fabric,
        "care": care,
        "department": department,
        "origin": origin,
        "closure": closure_types,
        "first_date": first_date,
        "rating": review,
        "star": star,
        "url": f"https://www.amazon.com/dp/{asin}"
    }
    return parsed
count = 0
extracted_data = {}
fail_dress_asin = []
remain_asin = get_remain_asin(PRODUCT_INFO_FILE, FILTERED_PRODUCT_ASIN_FILE)
async def scrape_product(asin: str, attribute: list, limit: httpx.Limits, timeout: httpx.Timeout, headers: dict, semaphore: asyncio.Semaphore) -> ProductInfo:
    """Scrape a single product page with randomized user-agent."""
    async with semaphore:
        global count
        user_agent = random.choice(USER_AGENTS)  # Rotate user agent for each request
        headers = {"user-agent": user_agent, **headers}
        
        try:
            async with httpx.AsyncClient(limits=limit, timeout=timeout, headers=headers) as client:
                response = await client.get(f"https://www.amazon.com/dp/{asin}")
                await asyncio.sleep(1)
                parsed_data = parse_product(response, asin, attribute)
                
                if parsed_data['name'] == '':
                    fail_dress_asin.append(asin)
                else:
                    extracted_data[asin] = parsed_data
                    print("Success")
                    # Save result immediately after scraping
                    with open(PRODUCT_INFO_FILE, 'a') as output_file:
                        json.dump(parsed_data, output_file, indent=4)
                        output_file.write('\n')  # Ensure each entry is written on a new line
                
                count += 1
                print(f"Count: {count}")
                return parsed_data
        except Exception as e:
            print(f"Error scraping {asin}: {e}")
            fail_dress_asin.append(asin)
            return {"asin": asin, "error": str(e)}

async def run():
    BASE_HEADERS = {
        "accept-language": "en-US,en;q=0.9",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br",
    }


    
    limits = httpx.Limits(max_connections=12)
    semaphore = asyncio.Semaphore(12)  # Limit to 12 concurrent requests
    
    tasks = []
    
    # with open(FILTERED_PRODUCT_ASIN_FILE, 'r') as file:
    #     data = json.load(file)
    for asin, attributes in remain_asin.items():
        task = asyncio.create_task(scrape_product(asin=asin, attribute = attributes, limit=limits, timeout=httpx.Timeout(15.0), headers=BASE_HEADERS, semaphore=semaphore))
        tasks.append(task)
        
    # Run all the tasks concurrently
    await asyncio.gather(*tasks)

    # Write failed ASINs to file
    with open(FAIL_TO_LOAD_ASIN_FILE, 'w') as output_file:
        json.dump(fail_dress_asin, output_file, indent=4)

if __name__ == "__main__":
    asyncio.run(run())