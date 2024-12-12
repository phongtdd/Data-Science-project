import asyncio
import httpx
import random
import json
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
   "Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 5.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.0.0 Safari/537.36",
    # Add more user agents as needed
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
                    print(f"Success for {asin}")
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
    
    with open(FILTERED_PRODUCT_ASIN_FILE, 'r') as file:
        data = json.load(file)
    for asin, attributes in data.items():
        task = asyncio.create_task(scrape_product(asin=asin, attribute=attributes, limit=limits, timeout=httpx.Timeout(15.0), headers=BASE_HEADERS, semaphore=semaphore))
        tasks.append(task)
    
    # Run all the tasks concurrently
    await asyncio.gather(*tasks)

    # Write failed ASINs to file
    with open(FAIL_TO_LOAD_ASIN_FILE, 'w') as output_file:
        json.dump(fail_dress_asin, output_file, indent=4)

if __name__ == "__main__":
    asyncio.run(run())
