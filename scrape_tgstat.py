from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime

def setup_driver():
    """Setup Chrome driver with appropriate options"""
    chrome_options = Options()
    # Add more realistic browser settings
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(options=chrome_options)

def get_channels_or_groups(driver, country_code, peer_type):
    """
    Navigate to TGStat and extract channel/group information
    country_code: 'ru' for Russia, 'ua' for Ukraine
    peer_type: 'channel' or 'chat'
    """
    try:
        # Navigate to the base URL
        driver.get("https://tgstat.ru/en/news")
        
        # Wait for Cloudflare verification
        print("Waiting for Cloudflare verification...")
        time.sleep(10)  # Give time for manual verification if needed
        
        # Wait for the page to be fully loaded
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "category-items-list-container"))
        )
        
        # Click the country dropdown
        country_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-light.border.dropdown-toggle"))
        )
        country_button.click()
        time.sleep(2)

        # Select the country
        country_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f"//img[@src='/img/flags/{country_code}.jpg']/.."))
        )
        country_option.click()
        time.sleep(2)

        # Click the peer type button (channel/group)
        peer_type_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"label.btn.btn-sm.btn-outline-dark.py-05.text-truncate.form-filter-js input[value='{peer_type}']"))
        )
        peer_type_label.click()
        time.sleep(2)

        # Get the CSRF token
        csrf_token = driver.find_element(By.NAME, "_tgstat_csrk").get_attribute("value")

        # Submit the form
        form = driver.find_element(By.ID, "category-list-form")
        driver.execute_script("arguments[0].submit();", form)
        time.sleep(5)  # Wait longer for the form submission to complete

        # Extract channel/group information
        items = []
        channel_elements = driver.find_elements(By.CSS_SELECTOR, "div.peer-item-box")
        
        for element in channel_elements:
            try:
                # Get the title
                title = element.find_element(By.CSS_SELECTOR, "div.font-16.text-dark.text-truncate").text.strip()
                
                # Get the link and username
                link = element.find_element(By.CSS_SELECTOR, "a[href*='/channel/']").get_attribute('href')
                username = link.split('/')[-1]
                
                # Get subscriber count
                subscribers = element.find_element(By.CSS_SELECTOR, "div.font-12.text-truncate b").text.strip()
                
                items.append({
                    'title': title,
                    'username': username,
                    'subscribers': subscribers
                })
            except Exception as e:
                print(f"Error extracting item: {str(e)}")
                continue

        return items

    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        return []

def save_to_file(items, filename):
    """Save items to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for item in items:
            f.write(f"Title: {item['title']}\n")
            f.write(f"Username: {item['username']}\n")
            f.write(f"Subscribers: {item['subscribers']}\n")
            f.write("-" * 50 + "\n\n")

def main():
    driver = setup_driver()
    try:
        # Define combinations to scrape
        combinations = [
            ("ru", "channel", "russian_channels.txt"),
            ("ru", "chat", "russian_groups.txt"),
            ("ua", "channel", "ukrainian_channels.txt"),
            ("ua", "chat", "ukrainian_groups.txt")
        ]
        
        for country, peer_type, filename in combinations:
            print(f"Fetching {country} {peer_type}...")
            items = get_channels_or_groups(driver, country, peer_type)
            if items:
                save_to_file(items, filename)
                print(f"Saved {len(items)} items to {filename}")
            time.sleep(5)  # Longer delay between requests
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main() 