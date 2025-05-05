import re
import os

# Get all files in the tgstat/html directory
input_dir = "data/telegram/tgstat/html"
output_dir = "data/telegram/channelsGroupsOfInterest"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Read the HTML file
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            html = f.read()

        # Find all tgstat URLs (both channels and chats, both domains)
        tgstat_links = re.findall(r'href="(https://(?:uk\.)?tgstat\.(?:ru|com)/en/(?:channel|chat)/[^"]+)"', html)

        # Convert each to the direct Telegram link and remove @
        tg_links = [
            link.replace("https://tgstat.ru/en/channel/@", "https://t.me/")
              .replace("https://tgstat.ru/en/chat/@", "https://t.me/")
              .replace("https://uk.tgstat.com/en/channel/@", "https://t.me/")
              .replace("https://uk.tgstat.com/en/chat/@", "https://t.me/")
            for link in tgstat_links
        ]
        
        # Filter for valid t.me links
        tg_links = [link for link in tg_links if link.startswith("https://t.me/")]

        # Create output filename
        output_filename = filename.replace('_html.txt', '_links.txt')
        
        # Write links to output file
        with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as out:
            for link in tg_links:
                out.write(link + "\n")

        print(f"Processed {filename}: Found {len(tg_links)} links")
