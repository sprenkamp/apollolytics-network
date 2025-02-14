import os
import pandas as pd
import asyncio
from sqlalchemy import create_engine
from telethon import TelegramClient
from telethon.errors import UsernameNotOccupiedError, UserIdInvalidError

# Load API credentials from environment variables
api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

if not api_id or not api_hash:
    raise ValueError("API ID and API Hash must be set as environment variables")

api_id = int(api_id)  # Convert to integer after validation

# Initialize the Telegram client
client = TelegramClient('session_name', api_id, api_hash)

# Database connection setup
connection_string = 'postgresql+psycopg2://kiliansprenkamp@localhost:5432/telegram_scraper'
engine = create_engine(connection_string)

table_name = "messages_community"
excluded_column = "embedding_user_bge_m3_32_bits"

def load_data():
    """
    Load data from the PostgreSQL database, excluding the specified column.
    """
    with engine.connect() as conn:
        # Fetch all column names except the excluded one
        columns_query = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
              AND table_schema = 'public'
        """
        df_cols = pd.read_sql(columns_query, conn)
        columns = df_cols['column_name'].tolist()
        
        # Remove the excluded column if it exists
        if excluded_column in columns:
            columns.remove(excluded_column)
        
        columns_str = ", ".join(columns)
        
        # Fetch data
        query = f"SELECT {columns_str} FROM {table_name};"
        df = pd.read_sql(query, conn)
    
    return df

async def fetch_bot_status_from_chat(chat_name, client):
    """
    Fetch bot status for all participants in a specific chat.
    Returns a dictionary mapping user_id to is_bot status.
    """
    bot_status_dict = {}
    try:
        print(f"Fetching participants from chat: {chat_name}")
        async for user in client.iter_participants(chat_name):
            bot_status_dict[user.id] = user.bot
        print(f"Fetched bot statuses from chat: {chat_name}")
    except Exception as e:
        print(f"Error fetching participants from chat '{chat_name}': {e}")
    return bot_status_dict

async def main(df, client, target_chats):
    """
    Main function to fetch and map bot status from specific chats.
    """
    # Ensure 'from_id' is integer and drop NaNs
    if 'from_id' not in df.columns:
        raise ValueError("'from_id' column is missing from the DataFrame.")
    
    # Drop NaNs in 'from_id' and ensure integer type
    initial_count = len(df)
    df = df.dropna(subset=['from_id'])
    df['from_id'] = df['from_id'].astype(int)
    final_count = len(df)
    print(f"Dropped {initial_count - final_count} rows due to NaN 'from_id'.")
    
    unique_user_ids = df['from_id'].unique().tolist()
    print(f"Unique user IDs to fetch: {len(unique_user_ids)}")
    
    # Fetch bot statuses from all target chats
    bot_status_dict = {}
    for chat in target_chats:
        chat_bot_status = await fetch_bot_status_from_chat(chat, client)
        bot_status_dict.update(chat_bot_status)
    
    total_bots_found = sum(bot_status_dict.values())
    print(f"Total bots found across all chats: {total_bots_found}")
    
    # Map 'is_bot' based on bot_status
    df['is_bot'] = df['from_id'].map(bot_status_dict)
    
    # Fill any missing 'is_bot' values with False
    df['is_bot'] = df['is_bot'].fillna(False)
    
    return df

async def run():
    """
    Runner function to execute the scraping and bot status mapping.
    """
    df = load_data()
    print(f"Total messages loaded: {len(df)}")
    
    # Define your target chats
    target_chats = ["https://t.me/readovchat", "https://t.me/specchatZ"]
    
    # Filter for specific chats and sample 10,000 messages
    df_sample = df[df["chat_name"].isin(target_chats)]
    print(f"Sampled {len(df_sample)} messages from target chats.")
    
    async with client:
        updated_df = await main(df_sample, client, target_chats)
    
    return updated_df

if __name__ == "__main__":
    try:
        updated_df = asyncio.run(run())
        print("\nBot Counts per Chat:")
        bot_counts = updated_df.groupby('chat_name')['is_bot'].sum().reset_index()
        bot_counts.rename(columns={'is_bot': 'number_of_bot_messages'}, inplace=True)
        print(bot_counts)
        
        print("\nList of Bot IDs per Chat:")
        bots_per_chat = updated_df[updated_df['is_bot']].groupby('chat_name')['from_id'].unique().reset_index()
        bots_per_chat.rename(columns={'from_id': 'bot_ids'}, inplace=True)
        print(bots_per_chat)
        
    except KeyboardInterrupt:
        print("Script interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
