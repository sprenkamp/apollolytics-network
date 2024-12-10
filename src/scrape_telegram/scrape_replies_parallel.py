import pandas as pd
import asyncio
import os
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon import types
from tqdm import tqdm 
import datetime
from multiprocessing import Manager
import time

load_dotenv()

def create_record(chat, message):
    record = dict()

    record['chat'] = chat

    record['id'] = message.id
    record['peer_id'] = message.peer_id.channel_id
    record['messageDatetime'] = message.date
    record['messageDate'] = message.date.strftime("%Y-%m-%d")
    record['messageText'] = message.message

    record['out'] = message.out
    record['mentioned'] = message.mentioned
    record['media_unread'] = message.media_unread
    record['silent'] = message.silent
    record['post'] = message.post
    record['from_scheduled'] = message.from_scheduled
    record['legacy'] = message.legacy
    record['edit_hide'] = message.edit_hide
    record['pinned'] = message.pinned
    record['noforwards'] = message.noforwards
    record['invert_media'] = message.invert_media
    record['offline'] = message.offline
    record['from_id'] = message.from_id
    record['from_boosts_applied'] = message.from_boosts_applied
    record['saved_peer_id'] = message.saved_peer_id
    
    fwd_from = None
    fwd_type = None
    if message.fwd_from is not None:
        if type(message.fwd_from.from_id) == types.PeerUser and message.fwd_from.from_id is not None:
            fwd_from = message.fwd_from.from_id.user_id
            fwd_type = 'user'
        elif message.fwd_from.from_id is not None:
            fwd_from = message.fwd_from.from_id.channel_id
            fwd_type = 'channel'

    record['fwd_from'] = fwd_from # only save channel ID of origin (is it even a channel or just the user?)
    record['fwd_from_type'] = fwd_type

    record['via_bot_id'] = message.via_bot_id
    record['via_business_bot_id'] = message.via_business_bot_id
    record['reply_to'] = message.reply_to
    # record['media'] = message.media #too many nested values related to images
    record['reply_markup'] = message.reply_markup
    record['entities'] = message.entities
    record['edit_date'] = message.edit_date
    record['post_author'] = message.post_author
    record['grouped_id'] = message.grouped_id
    record['restriction_reason'] = message.restriction_reason
    record['ttl_period'] = message.ttl_period
    record['quick_reply_shortcut_id'] = message.quick_reply_shortcut_id
    record['effect'] = message.effect
    record['factcheck'] = message.factcheck

    record['views'] = message.views if message.views is not None else 0
    record['forwards'] = message.forwards if message.forwards is not None else 0

    record['replies'] = message.replies.replies if message.replies else 0

    if message.reactions is None:
        record['reactions'] = {}
    else:
        reaction = {}
        for i in message.reactions.results:
            try:
                reaction[i.reaction.emoticon] = i.count
            except:
                pass
        record['reactions'] = reaction
    
    return record

async def scrape_chat(chat, message_id, client, result_list, semaphore):
    """
    This function asynchronously scrapes messages from a single chat.
    """
    # print(f'scraping {chat}/{message_id}')
    async with semaphore:
        data_list = []
        try:
            async for reply in client.iter_messages(chat, reply_to=message_id):
                if reply.message is not None and reply.message != '' and reply.message != 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.':
                    reply_record = create_record(chat, reply)
                    data_list.append(reply_record)
        except:
            x = 1
        # if len(data_list)>0:
            # print(f'Finished scraping {chat}/{message_id} with {len(data_list)} new replies')    
        result_list.append(data_list)

async def callAPI(file):
    """
    This function takes an input file and output CSV file path.
    It reads the input file, extracts the chats, and then asynchronously scrapes messages from each chat.
    Then it creates a dataframe from the scraped data and saves it to a CSV file after each chat.

    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_csv_path: path where the output CSV file will be saved containing the scraped data
    """

    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

    # Read existing CSV if it exists and extract the highest date for each chat
    if os.path.exists(file):
        df = pd.read_csv(file)
        # Group by chat and find the max date for each chat
        df['messageDatetime'] = pd.to_datetime(df['messageDatetime'])

    else:
        print(f"No file named {file} in path...")
        return None

    
    for c in tqdm(list(df.chat.unique()), desc=f"Chats progress"):
        all_data = Manager().list()
        output_csv_path = file.split('.')[0]+f'_replies_{c.split("/")[3]}.csv'
        if not os.path.exists(output_csv_path):
            async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
                # Create list of tasks
                tasks = []
                semaphore = asyncio.Semaphore(500)
                for message_id in list(df[df['chat']==c]['id'].values):
                    task = scrape_chat(c, message_id, client, all_data, semaphore)
                    tasks.append(task)

                print("Created list of all tasks")

                # tqdm for progress over chats
                with tqdm(total=len(tasks), desc=f"Scraping replies for {c}") as pbar:
                    for task in asyncio.as_completed(tasks):
                        await task  # Wait for each task to complete
                        pbar.update(1)  # Update progress bar

                        # Convert the result to DataFrame and save it to CSV
                        if len(all_data) > 0:
                            df_new = pd.DataFrame([record for sublist in all_data for record in sublist])
                            all_data[:] = []  # Clear the list

                            # Append to CSV file
                            
                            if not df_new.empty:
                                if os.path.exists(output_csv_path):
                                    df_new.to_csv(output_csv_path, mode='a', header=False, index=False)
                                else:
                                    df_new.to_csv(output_csv_path, index=False)
                client.disconnect()
            time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Chat Replies Scraper')
    parser.add_argument('--file', type=str, help='Path to the input file containing the list of chats to scrape')
    args = parser.parse_args()

    asyncio.run(callAPI(args.file))
