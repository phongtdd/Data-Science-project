import asyncio
import httpx
import re
import json
import random
from typing_extensions import TypedDict
from parsel import Selector
from typing import List
import time
from search import SEARCH_QUERY

FILTERED_PRODUCT_ASIN_FILE = f"data/{SEARCH_QUERY}/filtered_{SEARCH_QUERY}_asin.json"
FAIL_TO_LOAD_ASIN_FILE = f"data/{SEARCH_QUERY}/fail_{SEARCH_QUERY}_asin.json"
PRODUCT_INFO_FILE = f"data/{SEARCH_QUERY}/{SEARCH_QUERY}_info.json"

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
    # Add your user agents here
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/15.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edge/116.0.1938.69",
    "Mozilla/5.0 (Linux; Android 12; Pixel 6 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Mobile Safari/537.36 SamsungBrowser/16.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:92.0) Gecko/20100101 Firefox/92.0",
    "Mozilla/5.0 (Linux; Android 13; SM-A535F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.102 Mobile Safari/537.36 SamsungBrowser/19.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.91 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:93.0) Gecko/20100101 Firefox/93.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; SM-A505FN) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 9; SAMSUNG SM-J730F) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/11.1 Chrome/73.0.3683.90 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15"
]

# Function to extract keys (used later in `run`)
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

    # the other fields can be extracted with simple css selectors
    # we can define our helper functions to keep our code clean
    sel = Selector(text=result.text)

    # extract details from "Product Information" table:
    info_table = {}
    department = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Department')]/following-sibling::span/text()").get()
    care = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Care instructions')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if care!=None:
        care = care.strip()
    else:
        care = "None"
    fabric = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Fabric type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if fabric!=None:
        fabric = fabric.strip()
    else:
        fabric = "None"
    origin = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Origin')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if origin!=None:
        origin = origin.strip()
    else:
        origin = "None"    
    closure_types = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Closure type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get()
    if closure_types!=None:
        closure_types = closure_types.strip()
    else:
        closure_types = "None"      
    first_date = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Date First Available')]/following-sibling::span/text()").get()

    box = sel.css("#centerCol")
    
    brand = box.xpath("//*[@id='bylineInfo']/text()").get()
    
    price = box.css(".a-price.a-text-price.a-size-medium.apexPriceToPay > span.a-offscreen::text").get()
    # size = box.xpath(f"//*[@id='native_dropdown_selected_size_name']/option[@value!='-1' and contains(@value,'{asin}')]/text()").get()
    # if size!=None:
    #     size = size.strip()
    # else:
    #     size = "Select"
    # color = box.xpath("//*[@id='variation_color_name']/div/span/text()").get()
    # if color!=None:
    #     color = color.strip()
    # else:
    #     color = "None"
    review = sel.xpath("//*[@id='acrCustomerReviewText']/text()").get()
    star = sel.xpath("//*[@id='acrPopover']/span[1]/a/span/text()").get()
    if star!=None:
        star = star.strip()
    else:
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
async def scrape_product(asin: str,attribute: list, limit: httpx.Limits, timeout: httpx.Timeout, headers: dict, semaphore: asyncio.Semaphore) -> ProductInfo:
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
                if parsed_data['name']=='':
                    fail_dress_asin.append(asin)
                else:
                    extracted_data[asin] = parsed_data
                    print("Success")
                count+=1
                print(count)
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

    with open(FAIL_TO_LOAD_ASIN_FILE, "r") as file:
        all_asin = json.load(file)
    
    limits = httpx.Limits(max_connections=12)
    semaphore = asyncio.Semaphore(12)  # Limit to 8 concurrent requests
    
    tasks = []
    
    with open(FILTERED_PRODUCT_ASIN_FILE,'r') as file:
        data = json.load(file)
    for asin, attributes in data.items():
        task = asyncio.create_task(scrape_product(asin=asin, attribute = attributes, limit=limits, timeout=httpx.Timeout(15.0), headers=BASE_HEADERS, semaphore=semaphore))
        tasks.append(task)
    

    
    # Run all the tasks concurrently
    await asyncio.gather(*tasks)
    print(extracted_data)
    with open(PRODUCT_INFO_FILE, 'a') as output_file:
        # Write the parsed data
        json.dump(extracted_data, output_file, indent=4)
    with open(FAIL_TO_LOAD_ASIN_FILE, 'w') as output_file:
        # Write the parsed data
        json.dump(fail_dress_asin, output_file, indent=4)

if __name__ == "__main__":
    asyncio.run(run())


