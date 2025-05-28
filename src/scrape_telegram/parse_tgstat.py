import pandas as pd
from bs4 import BeautifulSoup
import os
import re

def is_valid_username(username):
    """Check if username is valid (not a random alphanumeric string)"""
    # Filter out usernames containing uppercase letters
    # These are typically random strings like: 8lFELgo6HTk4NTEy, AAAAAFQdoNVarTLkfTf06g, etc.
    return not any(c.isupper() for c in username)

def extract_channel_info(card):
    """Extract channel information from a card element"""
    try:
        # Get channel name
        name = card.find('div', class_='font-16').text.strip()
        
        # Get description
        desc = card.find('div', class_='font-14').text.strip()
        
        # Get subscriber/participant count - handle numbers with spaces
        sub_count = card.find('div', class_='font-12').find('b').text.strip()
        sub_count = int(sub_count.replace(' ', ''))  # Remove spaces before converting to int
        
        # Get channel URL and username from the href attribute
        link = card.find('a', href=True, class_='text-body')
        if not link:
            return None
            
        href = link['href']
        
        # Handle both channel and group URLs
        if '/chat/' in href:  # Group URL
            if '/id' in href:
                # For numeric IDs, we need to use the ID directly
                username = href.split('/id')[-1]
            else:
                # For usernames, remove the @ symbol
                username = href.split('/')[-1].replace('@', '')
        else:  # Channel URL
            username = href.split('/')[-1].replace('@', '')
            
        # Skip if username is invalid
        if not is_valid_username(username):
            return None
            
        tme_link = f"https://t.me/{username}"
        
        # Get last post time
        last_post = card.find('div', class_='text-center text-muted font-12')
        last_post = last_post.text.strip() if last_post else 'N/A'
        
        return {
            'name': name,
            'description': desc,
            'subscribers': sub_count,
            'url': href,
            'username': username,
            'tme_link': tme_link,
            'last_post': last_post
        }
    except Exception as e:
        print(f"Error extracting channel info: {e}")
        return None

def process_html_file(file_path, output_dir):
    """Process a single HTML file and create its outputs"""
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Read and parse HTML
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Extract channel information
    channels = []
    for card in soup.find_all('div', class_='peer-item-box'):
        channel_info = extract_channel_info(card)
        if channel_info:
            channels.append(channel_info)
    
    if not channels:
        print(f"No channels found in {os.path.basename(file_path)}")
        return
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(channels)
    df = df.sort_values('subscribers', ascending=False)
    
    # Generate base name for files
    base_name = os.path.basename(file_path).replace('_html.txt', '')
    
    # Save sorted channels to CSV
    csv_path = os.path.join(output_dir, f'{base_name}_sorted.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save top 50 t.me links to text file
    txt_path = os.path.join(output_dir, f'{base_name}_sorted.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for link in df.head(50)['tme_link']:
            f.write(f"{link}\n")
    
    # Print summary for this file
    print(f"\nSummary for {base_name}:")
    print(f"Total channels found: {len(df)}")
    print(f"Top 5 channels by subscribers:")
    for _, row in df.head().iterrows():
        print(f"- {row['name']}: {row['subscribers']:,} subscribers")
    print(f"Output files:")
    print(f"- CSV: {csv_path}")
    print(f"- TXT: {txt_path} (top 50 channels)\n")

def scrape_tgstat_files():
    """Scrape and sort channels from tgstat HTML files"""
    # Input and output directories
    html_dir = 'data/telegram/tgstat/html'
    output_dir = 'data/telegram/tgstat/sorted'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each HTML file separately
    for filename in os.listdir(html_dir):
        if filename.endswith('_html.txt'):
            file_path = os.path.join(html_dir, filename)
            process_html_file(file_path, output_dir)

if __name__ == "__main__":
    scrape_tgstat_files() 