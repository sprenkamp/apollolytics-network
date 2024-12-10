import pandas as pd
import asyncio
import os
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient, types
import datetime
import time
from tqdm import tqdm  # Import tqdm for progress bars
import sqlite3  # Import sqlite3 for SQLite database operations
import json  # Import json for better serialization
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from embedding.manage_embeddings import EmbeddingSystem

embedding_system = EmbeddingSystem()

load_dotenv()

# Define the SQLite database schema
CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id INTEGER,          -- Renamed from 'chat' to 'chat_id'
    chat_name TEXT,           -- New field for chat name (input identifier)
    id INTEGER PRIMARY KEY,
    peer_id INTEGER,
    messageDatetime TEXT,
    messageDate TEXT,
    messageText TEXT,
    out BOOLEAN,
    mentioned BOOLEAN,
    media_unread BOOLEAN,
    silent BOOLEAN,
    post BOOLEAN,
    from_scheduled BOOLEAN,
    legacy BOOLEAN,
    edit_hide BOOLEAN,
    pinned BOOLEAN,
    noforwards BOOLEAN,
    invert_media BOOLEAN,
    offline BOOLEAN,
    from_id INTEGER,
    from_boosts_applied INTEGER,
    saved_peer_id INTEGER,
    fwd_from INTEGER,
    fwd_from_type TEXT,
    via_bot_id INTEGER,
    via_business_bot_id INTEGER,
    reply_to INTEGER,
    reply_markup TEXT,
    entities TEXT,
    edit_date TEXT,
    post_author TEXT,
    grouped_id INTEGER,
    restriction_reason TEXT,
    ttl_period INTEGER,
    quick_reply_shortcut_id INTEGER,
    effect TEXT,
    factcheck TEXT,
    views INTEGER,
    forwards INTEGER,
    replies INTEGER,
    reactions TEXT,
    embeddding TEXT,         
    UNIQUE(chat_id, id) -- Ensure that each message is unique per chat
)
"""

INSERT_MESSAGE_QUERY = """
INSERT OR IGNORE INTO messages (
    chat_id, chat_name, id, peer_id, messageDatetime, messageDate, messageText, out, mentioned,
    media_unread, silent, post, from_scheduled, legacy, edit_hide, pinned, noforwards,
    invert_media, offline, from_id, from_boosts_applied, saved_peer_id, fwd_from,
    fwd_from_type, via_bot_id, via_business_bot_id, reply_to, reply_markup, entities,
    edit_date, post_author, grouped_id, restriction_reason, ttl_period,
    quick_reply_shortcut_id, effect, factcheck, views, forwards, replies, reactions, embeddding
) VALUES (
    :chat_id, :chat_name, :id, :peer_id, :messageDatetime, :messageDate, :messageText, :out, :mentioned,
    :media_unread, :silent, :post, :from_scheduled, :legacy, :edit_hide, :pinned, :noforwards,
    :invert_media, :offline, :from_id, :from_boosts_applied, :saved_peer_id, :fwd_from,
    :fwd_from_type, :via_bot_id, :via_business_bot_id, :reply_to, :reply_markup, :entities,
    :edit_date, :post_author, :grouped_id, :restriction_reason, :ttl_period,
    :quick_reply_shortcut_id, :effect, :factcheck, :views, :forwards, :replies, :reactions, :embeddding
)
"""

BATCH_SIZE = 1000  # Number of messages per batch insert

async def process_message(message, chat_name):
    """
    Process a single message and return a dictionary of the desired fields, including chat_id and chat_name.
    """
    if (
        message.message is None
        or message.message.strip() == ''
        or message.message == 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.'
    ):
        return None

    record = {
        'chat_id': message.peer_id.channel_id if isinstance(message.peer_id, types.PeerChannel) else (
            message.peer_id.user_id if isinstance(message.peer_id, types.PeerUser) else None
        ),
        'chat_name': chat_name,  # Include chat_name as the input identifier
        'id': message.id,
        'peer_id': message.peer_id.channel_id if isinstance(message.peer_id, types.PeerChannel) else (
            getattr(message.peer_id, 'user_id', None) if isinstance(message.peer_id, types.PeerUser) else None
        ),
        'messageDatetime': message.date.isoformat(),
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
        'reply_to': message.reply_to.reply_to_msg_id if message.reply_to else None,  # Extract msg_id
        'reply_markup': str(message.reply_markup) if message.reply_markup else None,
        'entities': str(message.entities) if message.entities else None,
        'edit_date': message.edit_date.isoformat() if message.edit_date else None,
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
        'reactions': json.dumps({reaction.reaction.emoticon: reaction.count for reaction in message.reactions.results}) if message.reactions else None,
        'embeddding': embedding_system.get_embedding(message.message)
    }

    if message.fwd_from:
        if isinstance(message.fwd_from.from_id, types.PeerUser):
            record['fwd_from'] = message.fwd_from.from_id.user_id
            record['fwd_from_type'] = 'user'
        elif isinstance(message.fwd_from.from_id, types.PeerChannel):
            record['fwd_from'] = message.fwd_from.from_id.channel_id
            record['fwd_from_type'] = 'channel'

    return record

async def insert_batches(queue, db_conn):
    """
    Consumer coroutine that inserts message records into the SQLite database in batches.
    """
    batch = []
    while True:
        record = await queue.get()
        if record is None:
            # Sentinel received, insert any remaining records and exit
            if batch:
                try:
                    db_conn.executemany(INSERT_MESSAGE_QUERY, batch)
                    db_conn.commit()
                except Exception as e:
                    print(f"Error inserting final batch: {e}")
            break
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            try:
                db_conn.executemany(INSERT_MESSAGE_QUERY, batch)
                db_conn.commit()
                batch.clear()
            except Exception as e:
                print(f"Error inserting batch: {e}")
    print("Batch insertion coroutine has finished.")

async def scrape_chat(chat_name, client, highest_date, db_conn, semaphore, queue, max_concurrent_tasks=10):
    """
    Asynchronously scrape messages from a single chat with parallel message processing and a tqdm progress bar.
    """

    print(f'\nStarting to scrape chat: {chat_name} from {highest_date}')
    start_time = time.time()

    message_count = 0

    # Initialize semaphore
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Initialize tqdm progress bar with no total (unknown)
    progress_bar = tqdm(desc=f"Scraping {chat_name}", unit="msg", dynamic_ncols=True)

    async def process_and_enqueue_message(message):
        nonlocal message_count
        try:
            record = await process_message(message, chat_name)
            if record:
                await queue.put(record)
                message_count += 1
                progress_bar.update(1)  # Update the progress bar
        except Exception as e:
            print(f"Error processing message {message.id} in chat {chat_name}: {e}")
        finally:
            semaphore.release()

    async for message in client.iter_messages(chat_name, reverse=True, offset_date=highest_date):
        await semaphore.acquire()
        asyncio.create_task(process_and_enqueue_message(message))

    # Wait until all semaphore slots are released, meaning all tasks have been started
    await semaphore.acquire()
    semaphore.release()

    # Allow some time for remaining tasks to finish
    await asyncio.sleep(0.1)

    # Close the progress bar
    progress_bar.close()

    elapsed_time = time.time() - start_time
    print(f"Finished scraping chat: {chat_name} with {message_count} new messages in {elapsed_time:.2f} seconds.")

async def callAPI(input_file_path, output_db_path):
    """
    Reads the input file, extracts the chats, and asynchronously scrapes messages from each chat.
    Then inserts the scraped data into a SQLite database in batches of 1000 messages.
    
    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_db_path: path where the output SQLite database file will be saved containing the scraped data
    """

    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        print("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in your environment variables.")
        return

    # Connect to the SQLite database (it will be created if it doesn't exist)
    db_conn = sqlite3.connect(output_db_path)
    cursor = db_conn.cursor()
    cursor.execute(CREATE_TABLE_QUERY)
    db_conn.commit()

    # Read the list of chats to scrape
    with open(input_file_path, 'r', encoding='utf-8') as file:
        chats = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    # Initialize the Telethon client
    async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        # Iterate through chats sequentially
        for chat_name in chats:

            # Retrieve the highest message date for the chat from the database
            cursor.execute("SELECT MAX(messageDatetime) FROM messages WHERE chat_name = ?", (chat_name,))
            result = cursor.fetchone()
            highest_date = datetime.datetime.fromisoformat(result[0]) if result and result[0] else None

            # Initialize a semaphore for controlling concurrent message processing
            semaphore = asyncio.Semaphore(10000)  # Adjust the number based on your requirements

            # Create a queue for batch insertion
            queue = asyncio.Queue()

            # Start the batch insertion consumer coroutine
            insert_task = asyncio.create_task(insert_batches(queue, db_conn))

            # Scrape the chat and enqueue messages
            await scrape_chat(chat_name, client, highest_date, db_conn, semaphore, queue)

            # Signal the consumer to finish by putting a sentinel
            await queue.put(None)

            # Wait for the consumer to finish inserting
            await insert_task

    # Close the database connection
    db_conn.close()

    print("Finished scraping all chats.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telegram Chat Scraper')
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the input file containing the list of chats to scrape')
    parser.add_argument('-o', '--output_db', type=str, required=True,
                        help='Path to the output SQLite database file where the scraped data will be saved')
    args = parser.parse_args()

    asyncio.run(callAPI(args.input_file, args.output_db))
