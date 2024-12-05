import asyncio
import random
import json
from typing_extensions import TypedDict
from parsel import Selector
import time
import aiohttp
from aiohttp import ClientSession

PRODUCT_INFO_FILE = f"shirts_info.json"
FAIL_TO_LOAD_ASIN_FILE = f"fail_shirt_asin.json"
VARIANT = f"variant_shirt_asin.json"
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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:112.0) Gecko/20100101 Firefox/112.0",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.74 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edge/124.0.4895.30",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-A536W) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Mobile Safari/537.36 SamsungBrowser/19.0",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.91 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edge/124.0.4895.30",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-G996B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edge/122.0.4942.47",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


# Global variable for storing ASINs that failed to load
fail_dress_asin = []

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

def parse_product(result, asin) -> ProductInfo:
    """parse Amazon's product page (e.g. https://www.amazon.com/dp/B07KR2N2GF) for essential product data"""
    sel = Selector(text=result)

    # Extract details from "Product Information" table
    department = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Department')]/following-sibling::span/text()").get()
    care = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Care instructions')]/ancestor::div[1]/following-sibling::div//span/span/text()").get() or "None"
    fabric = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Fabric type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get() or "None"
    origin = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Origin')]/ancestor::div[1]/following-sibling::div//span/span/text()").get() or "None"
    closure_types = sel.xpath("//*[@id='productFactsDesktopExpander']//span/span[contains(text(), 'Closure type')]/ancestor::div[1]/following-sibling::div//span/span/text()").get() or "None"
    first_date = sel.xpath("//*[@id='detailBullets_feature_div']//span/span[contains(normalize-space(text()), 'Date First Available')]/following-sibling::span/text()").get()

    # Box element for other details
    box = sel.css("#centerCol")
    brand = box.xpath("//*[@id='bylineInfo']/text()").get()
    price = box.css(".a-price.a-text-price.a-size-medium.apexPriceToPay > span.a-offscreen::text").get()
    size = box.xpath(f"//*[@id='native_dropdown_selected_size_name']/option[@value!='-1' and contains(@value,'{asin}')]/text()").get() or "Select"
    color = box.xpath("//*[@id='variation_color_name']/div/span/text()").get() or "None"
    if not color:
        color = sel.css('.a-dropdown-prompt::text').get()  # Simulating JavaScript-like extraction of color
        if color:
            color = color.strip()
    else:
        color = color.strip() if color else "None"
    review = sel.xpath("//*[@id='acrCustomerReviewText']/text()").get()
    star = sel.xpath("//*[@id='acrPopover']/span[1]/a/span/text()").get() or "None"

    parsed = {
        "name": sel.css("#productTitle::text").get("").strip(),
        "asin": asin,
        "brand": brand,
        "color": color,
        "size": size,
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

async def scrape_product(asin: str, session: ClientSession, semaphore: asyncio.Semaphore) -> ProductInfo:
    """Scrape a single product page with randomized user-agent."""
    global count
    user_agent = random.choice(USER_AGENTS)  # Rotate user agent for each request
    headers = {"user-agent": user_agent}
    
    async with semaphore:
        try:
            async with session.get(f"https://www.amazon.com/dp/{asin}", headers=headers) as response:
                result = await response.text()
                await asyncio.sleep(1)
                parsed_data = parse_product(result, asin)
                if parsed_data['name'] == '':
                    fail_dress_asin.append(asin)
                else:
                    extracted_data[asin] = parsed_data
                    print("Success")
                    # Save the result immediately after scraping
                    with open(PRODUCT_INFO_FILE, 'a') as output_file:
                        json.dump(parsed_data, output_file, indent=4)
    # Write the opening square bracket if the file is empty
                        if output_file.tell() == 0:
                                output_file.write('[\n')
                        else:
                                output_file.write(',\n') # Add a newline to separate each entry
                count += 1
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

    with open(VARIANT, "r") as file:
        all_asin = json.load(file)
    print(len(all_asin))
    semaphore = asyncio.Semaphore(16)  # Limit to 8 concurrent requests
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for asin in all_asin:  # Adjust the range as necessary
            task = asyncio.create_task(scrape_product(asin=asin, session=session, semaphore=semaphore))
            tasks.append(task)
        
        # Run all the tasks concurrently
        await asyncio.gather(*tasks)
        print(extracted_data)
        
        # Save fail ASINs to a separate file at the end
        with open(FAIL_TO_LOAD_ASIN_FILE, 'w') as output_file:
            json.dump(fail_dress_asin, output_file, indent=4)

if __name__ == "__main__":
    asyncio.run(run())
