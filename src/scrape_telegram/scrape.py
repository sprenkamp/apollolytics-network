import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient
from tqdm import tqdm 
import datetime

load_dotenv() 

def scrape_chat(chat, client, output_csv_path):
    """
    This function synchronously scrapes messages from a single chat.
    """
    print('scraping chat:', chat)

    # Check if output_csv_path already exists and read its highest date
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        df_sub = df[df['chat'] == chat]
        highest_date = pd.to_datetime(df_sub['messageDatetime']).max()
        if pd.isna(highest_date):
            highest_date = datetime.datetime(2022, 2, 24) 
    else:
        highest_date = datetime.datetime(2022, 2, 24)
    print("highest_date:", highest_date)
    # find max time in the database
    max_time = highest_date  # Set max_time to the highest_date

    data_list = []

    for message in client.iter_messages(chat, reverse=True, offset_date=max_time):

        if message.message is not None and message.message != '':
            record = dict()
            record['chat'] = chat

            record['messageDatetime'] = message.date
            record['messageDate'] = message.date.strftime("%Y-%m-%d")   
            record['messageText'] = message.message

            record['views'] = message.views if message.views is not None else 0
            record['forwards'] = message.forwards if message.forwards is not None else 0

            if message.replies is None:
                record['replies'] = 0
            else:
                record['replies'] = message.replies.replies

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

    print("data len:", len(data_list))
    return data_list

def callAPI(input_file_path, output_csv_path):
    """
    This function takes an input file and output CSV file path.
    It reads the input file, extracts the chats, and then synchronously scrapes messages from each chat.
    Then it creates a dataframe from the scraped data and saves it to a CSV file.

    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_csv_path: path where the output CSV file will be saved containing the scraped data
    """
    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

    with open(input_file_path) as file:
        chats = file.readlines()
        chats = [chat.strip() for chat in chats if not chat.startswith("#")]

    all_data = []

    with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        for chat in tqdm(chats):
            data_list = scrape_chat(chat, client, output_csv_path)
            if len(data_list) > 0:
                df_new = pd.DataFrame(data_list)
                # If output_csv_path already exists, append to it; otherwise, create a new file
                if os.path.exists(output_csv_path):
                    df = pd.read_csv(output_csv_path)
                    df_concat = pd.concat([df, df_new])
                    df_concat.to_csv(output_csv_path, index=False)
                else:
                    df_new.to_csv(output_csv_path, index=False)
                print("Data saved to CSV:", output_csv_path)
            else:
                print("No data scraped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Chat Scraper')
    parser.add_argument('--input_file', type=str, help='Path to the input file containing the list of chats to scrape')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file where the scraped data will be saved')
    args = parser.parse_args()

    callAPI(args.input_file, args.output_file)
