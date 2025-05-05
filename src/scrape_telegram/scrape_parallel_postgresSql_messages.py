import asyncio
import os
import argparse
import datetime
import time
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient, types
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.channels import GetFullChannelRequest
from tqdm.asyncio import tqdm
import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("telegram_scraper.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection details from environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "kiliansprenkamp")  # Changed from UZH to your system username
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # Empty if not set

# Telegram API credentials from environment variables
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
    logging.error("Please set TELEGRAM_API_ID and TELEGRAM_API_HASH in your environment variables.")
    exit(1)

# Define table names for different categories
TABLE_NAMES = {
    'ru_channels': 'russian_channels_messages',
    'ru_groups': 'russian_groups_messages',
    'ua_channels': 'ukrainian_channels_messages',
    'ua_groups': 'ukrainian_groups_messages'
}

# Define the PostgreSQL database schema with composite primary key (chat_id, id)
def get_create_table_query(table_name):
    return f"""
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE IF NOT EXISTS {table_name} (
        chat_id BIGINT NOT NULL,
        id BIGINT NOT NULL,
        chat_name TEXT,
        peer_id BIGINT,
        messageDatetime TIMESTAMP,
        messageDate DATE,
        messageText TEXT,
        "out" BOOLEAN,
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
        from_id BIGINT,
        from_boosts_applied INTEGER,
        saved_peer_id BIGINT,
        fwd_from BIGINT,
        fwd_from_type TEXT,
        via_bot_id BIGINT,
        via_business_bot_id BIGINT,
        reply_to BIGINT,
        reply_markup TEXT,
        entities TEXT,
        edit_date TIMESTAMP,
        post_author TEXT,
        grouped_id BIGINT,
        ttl_period INTEGER,
        quick_reply_shortcut_id BIGINT,
        effect TEXT,
        factcheck TEXT,
        views INTEGER,
        forwards INTEGER,
        replies INTEGER,
        reactions JSONB,
        embedding TEXT,
        PRIMARY KEY (chat_id, id)
    );
    """

def get_insert_query(table_name):
    return f"""
    INSERT INTO {table_name} (
        chat_id, id, chat_name, peer_id, messageDatetime, messageDate, messageText, "out", mentioned,
        media_unread, silent, post, from_scheduled, legacy, edit_hide, pinned, noforwards,
        invert_media, offline, from_id, from_boosts_applied, saved_peer_id, fwd_from,
        fwd_from_type, via_bot_id, via_business_bot_id, reply_to, reply_markup, entities,
        edit_date, post_author, grouped_id, ttl_period,
        quick_reply_shortcut_id, effect, factcheck, views, forwards, replies, reactions, embedding
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9,
        $10, $11, $12, $13, $14, $15, $16, $17,
        $18, $19, $20, $21, $22, $23,
        $24, $25, $26, $27, $28, $29,
        $30, $31, $32, $33,
        $34, $35, $36, $37, $38, $39, $40, $41
    )
    ON CONFLICT (chat_id, id) DO NOTHING
    """

BATCH_SIZE = 25000  # Number of messages per batch insert

async def process_message(message, chat_name):
    """
    Process a single message and return a tuple of the desired fields, excluding restriction_reason.
    Handles missing 'emoticon' attributes gracefully by setting them to None.
    """
    if (
        message.message is None
        or message.message.strip() == ''
        or message.message == 'Невозможно подключиться к серверу. Проверьте соединение и повторите попытку.'
    ):
        return None

    # Extract and process fields
    chat_id = None
    if isinstance(message.peer_id, types.PeerChannel):
        chat_id = message.peer_id.channel_id
    elif isinstance(message.peer_id, types.PeerUser):
        chat_id = message.peer_id.user_id

    message_id = message.id
    peer_id = chat_id  # Assuming peer_id is similar to chat_id

    message_datetime = message.date.replace(tzinfo=None)
    message_date = message.date.date()
    message_text = message.message
    out = message.out
    mentioned = message.mentioned
    media_unread = message.media_unread
    silent = message.silent
    post = message.post
    from_scheduled = message.from_scheduled
    legacy = message.legacy
    edit_hide = message.edit_hide
    pinned = message.pinned
    noforwards = message.noforwards
    invert_media = message.invert_media
    offline = message.offline

    from_id = None
    if isinstance(message.from_id, types.PeerUser):
        from_id = message.from_id.user_id
    elif isinstance(message.from_id, types.PeerChannel):
        from_id = message.from_id.channel_id

    from_boosts_applied = message.from_boosts_applied
    saved_peer_id = message.saved_peer_id
    fwd_from = None
    fwd_from_type = None
    if message.fwd_from:
        if isinstance(message.fwd_from.from_id, types.PeerUser):
            fwd_from = message.fwd_from.from_id.user_id
            fwd_from_type = 'user'
        elif isinstance(message.fwd_from.from_id, types.PeerChannel):
            fwd_from = message.fwd_from.from_id.channel_id
            fwd_from_type = 'channel'

    via_bot_id = message.via_bot_id
    via_business_bot_id = message.via_business_bot_id
    reply_to = message.reply_to.reply_to_msg_id if message.reply_to else None
    reply_markup = str(message.reply_markup) if message.reply_markup else None
    entities = str(message.entities) if message.entities else None
    edit_date = message.edit_date.replace(tzinfo=None) if message.edit_date else None
    post_author = message.post_author
    grouped_id = message.grouped_id
    ttl_period = message.ttl_period
    quick_reply_shortcut_id = message.quick_reply_shortcut_id
    effect = message.effect
    factcheck = message.factcheck
    views = message.views if message.views is not None else 0
    forwards = message.forwards if message.forwards is not None else 0
    replies = message.replies.replies if message.replies else 0

    # Handle Reactions Safely
    reactions = {}
    if hasattr(message, 'reactions') and message.reactions:
        for reaction in message.reactions.results:
            emoticon = getattr(reaction.reaction, 'emoticon', None)
            # You can choose to skip or include reactions without emoticon
            if emoticon is not None:
                reactions[emoticon] = reaction.count
            else:
                # Option 1: Skip reactions without emoticon
                # continue

                # Option 2: Include with None or a placeholder
                reactions[None] = reaction.count  # Using None as key

                # Option 3: Use a placeholder string
                # reactions['N/A'] = reaction.count

    reactions_json = json.dumps(reactions) if reactions else None
    embedding = None  # Assuming embedding is handled elsewhere

    record = (
        chat_id,           # chat_id ($1)
        message_id,        # id ($2)
        chat_name,         # chat_name ($3)
        peer_id,           # peer_id ($4)
        message_datetime,  # messageDatetime ($5)
        message_date,      # messageDate ($6)
        message_text,      # messageText ($7)
        out,               # out ($8)
        mentioned,         # mentioned ($9)
        media_unread,      # media_unread ($10)
        silent,            # silent ($11)
        post,              # post ($12)
        from_scheduled,    # from_scheduled ($13)
        legacy,            # legacy ($14)
        edit_hide,         # edit_hide ($15)
        pinned,            # pinned ($16)
        noforwards,        # noforwards ($17)
        invert_media,      # invert_media ($18)
        offline,           # offline ($19)
        from_id,           # from_id ($20)
        from_boosts_applied,  # from_boosts_applied ($21)
        saved_peer_id,         # saved_peer_id ($22)
        fwd_from,              # fwd_from ($23)
        fwd_from_type,         # fwd_from_type ($24)
        via_bot_id,            # via_bot_id ($25)
        via_business_bot_id,   # via_business_bot_id ($26)
        reply_to,              # reply_to ($27)
        reply_markup,          # reply_markup ($28)
        entities,              # entities ($29)
        edit_date,             # edit_date ($30)
        post_author,           # post_author ($31)
        grouped_id,            # grouped_id ($32)
        ttl_period,            # ttl_period ($33)
        quick_reply_shortcut_id,  # quick_reply_shortcut_id ($34)
        effect,                   # effect ($35)
        factcheck,                # factcheck ($36)
        views,                    # views ($37)
        forwards,                 # forwards ($38)
        replies,                  # replies ($39)
        reactions_json,           # reactions ($40)
        embedding                 # embedding ($41)
    )

    return record

async def insert_batches(queue, db_pool):
    """
    Consumer coroutine that inserts message records into the PostgreSQL database in batches.
    """
    batch = []
    while True:
        record = await queue.get()
        if record is None:
            # Sentinel received, insert any remaining records and exit
            if batch:
                try:
                    await db_pool.executemany(INSERT_MESSAGE_QUERY, batch)
                    logging.info(f"Inserted final batch of {len(batch)} records.")
                except Exception as e:
                    logging.error(f"Error inserting final batch: {e}")
            break
        batch.append(record)
        if len(batch) >= BATCH_SIZE:
            try:
                await db_pool.executemany(INSERT_MESSAGE_QUERY, batch)
                logging.info(f"Inserted batch of {len(batch)} records.")
            except Exception as e:
                logging.error(f"Error inserting batch: {e}")
            batch.clear()
    logging.info("Batch insertion coroutine has finished.")

async def scrape_chat(chat_name, client, highest_date, db_pool, queue, max_concurrent_tasks=10):
    """
    Asynchronously scrape messages from a single chat with parallel message processing and a tqdm progress bar.
    """
    logging.info(f'\nStarting to scrape chat: {chat_name} from {highest_date}')
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
            logging.error(f"Error processing message {message.id} in chat {chat_name}: {e}")
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
    logging.info(f"Finished scraping chat: {chat_name} with {message_count} new messages in {elapsed_time:.2f} seconds.")

async def callAPI(input_file_path):
    """
    Reads the input file, extracts the chats, and asynchronously scrapes messages from each chat.
    Then inserts the scraped data into a PostgreSQL database in batches of 25,000 messages.
    
    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    """
    # Read the list of chats to scrape
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            chats = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        logging.error(f"Input file '{input_file_path}' not found.")
        return
    except Exception as e:
        logging.error(f"Error reading input file '{input_file_path}': {e}")
        return

    # Create a connection pool to PostgreSQL
    try:
        db_pool = await asyncpg.create_pool(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD if POSTGRES_PASSWORD else None,
            min_size=1,
            max_size=10  # Adjust based on your requirements
        )
    except Exception as e:
        logging.error(f"Failed to create PostgreSQL connection pool: {e}")
        return

    # Create the messages table
    try:
        async with db_pool.acquire() as connection:
            await connection.execute(CREATE_TABLE_QUERY)
            logging.info("Ensured that the 'messages' table exists.")
    except Exception as e:
        logging.error(f"Failed to create table: {e}")
        await db_pool.close()
        return

    # Initialize the Telethon client
    try:
        async with TelegramClient('SessionName', TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
            # Iterate through chats sequentially
            for chat_name in chats:
                # Retrieve the chat entity to get chat_id
                try:
                    entity = await client.get_entity(chat_name)
                    chat_id = entity.id
                except Exception as e:
                    logging.error(f"Failed to get entity for chat '{chat_name}': {e}")
                    continue  # Skip this chat and move to the next

                # Retrieve the highest message date for the chat from the database
                try:
                    async with db_pool.acquire() as connection:
                        result = await connection.fetchrow(
                            "SELECT MAX(messageDatetime) FROM messages WHERE chat_id = $1", chat_id
                        )
                    # Set highest_date to result['max'] or default to January 1, 2021
                    highest_date = result['max'] if result and result['max'] else datetime.datetime(2021, 1, 1)
                except Exception as e:
                    logging.error(f"Error fetching highest_date for chat {chat_name}: {e}")
                    # If there's an error fetching the date, default to January 1, 2021
                    highest_date = datetime.datetime(2021, 1, 1)

                # Create a queue for batch insertion
                queue = asyncio.Queue()

                # Start the batch insertion consumer coroutine
                insert_task = asyncio.create_task(insert_batches(queue, db_pool))

                # Scrape the chat and enqueue messages
                await scrape_chat(chat_name, client, highest_date, db_pool, queue)

                # Signal the consumer to finish by putting a sentinel
                await queue.put(None)

                # Wait for the consumer to finish inserting
                await insert_task

    except Exception as e:
        logging.error(f"Error initializing Telegram client: {e}")
    finally:
        # Close the database pool
        await db_pool.close()

    logging.info("Finished scraping all chats.")

async def setup_database():
    """Create all necessary tables in the database"""
    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    
    try:
        for table_name in TABLE_NAMES.values():
            await conn.execute(get_create_table_query(table_name))
            logging.info(f"Created table {table_name}")
    finally:
        await conn.close()

async def process_file(file_path, table_name, client, db_pool, queue):
    """Process a single file of Telegram links"""
    with open(file_path, 'r') as f:
        chat_names = [line.strip().replace('https://t.me/', '') for line in f if line.strip()]
    
    tasks = []
    for chat_name in chat_names:
        try:
            # Check if the chat exists
            entity = await client.get_entity(chat_name)
            if entity:
                logging.info(f"Found chat: {chat_name}")
                task = asyncio.create_task(
                    scrape_chat(chat_name, client, None, db_pool, queue, table_name)
                )
                tasks.append(task)
            else:
                logging.warning(f"Chat not found: {chat_name}")
        except Exception as e:
            logging.error(f"Error checking chat {chat_name}: {e}")
            continue
    
    if tasks:
        await asyncio.gather(*tasks)
    else:
        logging.warning(f"No valid chats found in {file_path}")

async def main():
    # Setup database tables
    await setup_database()
    
    # Create database connection pool
    db_pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    
    # Create message queue
    queue = asyncio.Queue()
    
    # Start the batch inserter
    inserter = asyncio.create_task(insert_batches(queue, db_pool))
    
    # Initialize Telegram client
    client = TelegramClient('session_name', TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()
    
    try:
        # Process each file
        input_dir = Path("data/telegram/channelsGroupsOfInterest")
        for file_path in input_dir.glob("*_links.txt"):
            # Determine table name based on filename
            if "ru_channels" in file_path.name:
                table_name = TABLE_NAMES['ru_channels']
            elif "ru_groups" in file_path.name:
                table_name = TABLE_NAMES['ru_groups']
            elif "ua_channels" in file_path.name:
                table_name = TABLE_NAMES['ua_channels']
            elif "ua_groups" in file_path.name:
                table_name = TABLE_NAMES['ua_groups']
            else:
                continue
            
            logging.info(f"Processing {file_path} into table {table_name}")
            await process_file(file_path, table_name, client, db_pool, queue)
    
    finally:
        # Cleanup
        await queue.join()  # Wait for all messages to be processed
        inserter.cancel()  # Cancel the inserter task
        await db_pool.close()
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
