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

load_dotenv()

async def scrape_chat(chat, client, highest_date, result_list):
    """
    This function asynchronously scrapes messages from a single chat.
    """
    print('scraping chat:', chat)

    max_time = highest_date  # Set max_time to the highest_date
    data_list = []
    async for message in client.iter_messages(chat, reverse=True, offset_date=max_time):

        if message.message is not None and message.message != '' and message.message != 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.':
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

            data_list.append(record)

    print(f"Finished scraping chat: {chat} with {len(data_list)} new messages.")
    result_list.append(data_list)

async def callAPI(input_file_path, output_csv_path):
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
    highest_dates = {}
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        # Group by chat and find the max date for each chat
        df['messageDatetime'] = pd.to_datetime(df['messageDatetime'])
        highest_dates = df.groupby('chat')['messageDatetime'].max().to_dict()

    else:
        df = pd.DataFrame()  # Empty dataframe if no file exists

    with open(input_file_path) as file:
        chats = file.readlines()
        chats = [chat.strip() for chat in chats if not chat.startswith("#")]

    all_data = Manager().list()

    async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        # Create list of tasks
        tasks = []
        for chat in chats:
            highest_date = highest_dates.get(chat, None)  # Get highest date for chat
            task = scrape_chat(chat, client, highest_date, all_data)
            tasks.append(task)

        # tqdm for progress over chats
        with tqdm(total=len(tasks), desc="Scraping chats") as pbar:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Chat Scraper')
    parser.add_argument('-i', '--input_file', type=str, help='Path to the input file containing the list of chats to scrape')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output CSV file where the scraped data will be saved')
    args = parser.parse_args()

    asyncio.run(callAPI(args.input_file, args.output_file))
