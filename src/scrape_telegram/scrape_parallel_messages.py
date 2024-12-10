import pandas as pd
import asyncio
import os
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient, types
import datetime
import time
from tqdm import tqdm  # Import tqdm for progress bars

load_dotenv()

async def process_message(message):
    """
    Process a single message and return a dictionary of the desired fields.
    """
    if (
        message.message is None
        or message.message.strip() == ''
        or message.message == 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.'
    ):
        return None

    record = {
        'chat': message.peer_id.channel_id if isinstance(message.peer_id, types.PeerChannel) else (
            message.peer_id.user_id if isinstance(message.peer_id, types.PeerUser) else None
        ),
        'id': message.id,
        'peer_id': message.peer_id.channel_id if isinstance(message.peer_id, types.PeerChannel) else (
            getattr(message.peer_id, 'user_id', None) if isinstance(message.peer_id, types.PeerUser) else None
        ),
        'messageDatetime': message.date,
        'messageDate': message.date.strftime("%Y-%m-%d"),
        'messageText': message.message,
        'out': message.out,
        'mentioned': message.mentioned,
        'media_unread': message.media_unread,
        'silent': message.silent,
        'post': message.post,
        'from_scheduled': message.from_scheduled,
        'legacy': message.legacy,
        'edit_hide': message.edit_hide,
        'pinned': message.pinned,
        'noforwards': message.noforwards,
        'invert_media': message.invert_media,
        'offline': message.offline,
        'from_id': message.from_id.user_id if isinstance(message.from_id, types.PeerUser) else (
            getattr(message.from_id, 'channel_id', None) if isinstance(message.from_id, types.PeerChannel) else None
        ),
        'from_boosts_applied': message.from_boosts_applied,
        'saved_peer_id': message.saved_peer_id,
        'fwd_from': None,
        'fwd_from_type': None,
        'via_bot_id': message.via_bot_id,
        'via_business_bot_id': message.via_business_bot_id,
        'reply_to': message.reply_to,
        'reply_markup': message.reply_markup,
        'entities': message.entities,
        'edit_date': message.edit_date,
        'post_author': message.post_author,
        'grouped_id': message.grouped_id,
        'restriction_reason': message.restriction_reason,
        'ttl_period': message.ttl_period,
        'quick_reply_shortcut_id': message.quick_reply_shortcut_id,
        'effect': message.effect,
        'factcheck': message.factcheck,
        'views': message.views if message.views is not None else 0,
        'forwards': message.forwards if message.forwards is not None else 0,
        'replies': message.replies.replies if message.replies else 0,
        'reactions': {}
    }

    if message.fwd_from:
        if isinstance(message.fwd_from.from_id, types.PeerUser):
            record['fwd_from'] = message.fwd_from.from_id.user_id
            record['fwd_from_type'] = 'user'
        elif isinstance(message.fwd_from.from_id, types.PeerChannel):
            record['fwd_from'] = message.fwd_from.from_id.channel_id
            record['fwd_from_type'] = 'channel'

    if message.reactions:
        for reaction in message.reactions.results:
            try:
                emoticon = reaction.reaction.emoticon
                count = reaction.count
                record['reactions'][emoticon] = count
            except AttributeError:
                continue

    return record

async def scrape_chat(chat, client, highest_date, result_list, semaphore, max_concurrent_tasks=10):
    """
    Asynchronously scrape messages from a single chat with parallel message processing and a tqdm progress bar.
    """
    print(f'\nStarting to scrape chat: {chat}')
    start_time = time.time()

    data_list = []
    tasks = []
    message_count = 0

    # Initialize semaphore
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Initialize tqdm progress bar with no total (unknown)
    progress_bar = tqdm(desc=f"Scraping {chat}", unit="msg", dynamic_ncols=True)

    async def process_and_store_message(message):
        nonlocal message_count
        try:
            record = await process_message(message)
            if record:
                data_list.append(record)
                message_count += 1
                progress_bar.update(1)  # Update the progress bar
        except Exception as e:
            print(f"Error processing message {message.id} in chat {chat}: {e}")
        finally:
            semaphore.release()

    async for message in client.iter_messages(chat, reverse=True, offset_date=highest_date):
        await semaphore.acquire()
        task = asyncio.create_task(process_and_store_message(message))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Close the progress bar
    progress_bar.close()

    elapsed_time = time.time() - start_time
    print(f"Finished scraping chat: {chat} with {len(data_list)} new messages in {elapsed_time:.2f} seconds.")
    result_list.extend(data_list)

async def callAPI(input_file_path, output_csv_path):
    """
    Reads the input file, extracts the chats, and asynchronously scrapes messages from each chat.
    Then creates a dataframe from the scraped data and saves it to a CSV file after each chat.

    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_csv_path: path where the output CSV file will be saved containing the scraped data
    """

    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        print("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in your environment variables.")
        return

    # Read existing CSV if it exists and extract the highest date for each chat
    highest_dates = {}
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path, parse_dates=['messageDatetime'])
        # Group by chat and find the max date for each chat
        highest_dates = df.groupby('chat')['messageDatetime'].max().to_dict()
    else:
        df = pd.DataFrame()  # Empty dataframe if no file exists

    # Read the list of chats to scrape
    with open(input_file_path, 'r', encoding='utf-8') as file:
        chats = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    # Initialize the Telethon client
    async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        all_data = []  # List to store all scraped data

        # Iterate through chats sequentially
        for chat in chats:
            highest_date = highest_dates.get(chat, None)  # Get highest date for chat

            # Initialize a semaphore for controlling concurrent message processing
            semaphore = asyncio.Semaphore(25)  # Adjust the number based on your requirements

            await scrape_chat(chat, client, highest_date, all_data, semaphore)

            # After scraping each chat, save the data to CSV
            if all_data:
                df_new = pd.DataFrame(all_data)
                all_data = []  # Clear the list after saving

                # Append to CSV file
                if os.path.exists(output_csv_path):
                    df_new.to_csv(output_csv_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(output_csv_path, index=False)

    print("Finished scraping all chats.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Chat Scraper')
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the input file containing the list of chats to scrape')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Path to the output CSV file where the scraped data will be saved')
    args = parser.parse_args()

    asyncio.run(callAPI(args.input_file, args.output_file))
