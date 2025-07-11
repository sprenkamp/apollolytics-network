import os
import asyncio
import asyncpg
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import logging
import re
import math

# Configure logging
logging.basicConfig(
    filename='insertion_errors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# Load environment variables from .env file
load_dotenv()

# ==========================
# Configuration Variables
# ==========================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "your_username")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

BATCH_SIZE = 10000
TABLE_CONFIG = {
    "russian_channels": "RussianNarratives",
    "russian_groups": "RussianNarratives",
    "ukrainian_channels": "UkrainianNarratives",
    "ukrainian_groups": "UkrainianNarratives"
}
NARRATIVES_FILE_PATH = 'data/propaganda_narratives/propagandaNarratives.xlsx'

# ==========================
# Helper Functions
# ==========================

def sanitize_key(key):
    sanitized = re.sub(r'[ \-/]', '_', key)
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)
    return sanitized

def get_embedding(text):
    return MODEL.encode(text).tolist()

def load_narratives_from_excel(file_path):
    """Load narratives from an Excel file."""
    try:
        xls = pd.ExcelFile(file_path)
        narratives = {}
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            narratives[sheet_name] = dict(zip(df['NarrativeName'], df['NarrativeOriginalLanguage']))
        print(f"Successfully loaded narratives from {file_path}")
        return narratives
    except FileNotFoundError:
        logging.error(f"Narratives file not found at {file_path}")
        print(f"Narratives file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading narratives from Excel file: {e}")
        print(f"Error reading narratives from Excel file: {e}")
        return None

async def add_similarity_columns_if_not_exist(pool, table_name, narrative_keys):
    """
    Adds new columns for narrative similarities to an existing table if they don't already exist.
    """
    column_add_statements = []
    for key in narrative_keys:
        sanitized_key = sanitize_key(key)
        column_name = f"{sanitized_key}_similarity"
        # Using 'ADD COLUMN IF NOT EXISTS' which is safe and idempotent in PostgreSQL 9.6+
        column_add_statements.append(f"ADD COLUMN IF NOT EXISTS {column_name} FLOAT")

    if not column_add_statements:
        return

    alter_table_query = f"ALTER TABLE {table_name} {', '.join(column_add_statements)}"

    try:
        async with pool.acquire() as connection:
            await connection.execute(alter_table_query)
        print(f"Ensured similarity columns exist in table '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to add columns to table '{table_name}': {e}")
        print(f"Failed to add columns to table '{table_name}': {e}")

async def fetch_embeddings_keyset(connection, table_name, column_name, fetch_columns, batch_size=1000):
    last_id = 0
    while True:
        query = f"""
            SELECT {", ".join(fetch_columns)}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL AND id > $1
            ORDER BY id
            LIMIT {batch_size}
        """
        rows = await connection.fetch(query, last_id)
        if not rows:
            break
        yield rows
        last_id = rows[-1]['id']

async def process_table(table_key, narrative_sheet_name, all_narratives, batch_size=1000):
    source_table = f"{table_key}_messages"
    print(f"\n>>> Processing table: {source_table} with narratives from {narrative_sheet_name} <<<")
    embedding_column = "multilingual_e5_large_instruct"
    selected_read_columns = ["id", embedding_column]

    pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        min_size=1,
        max_size=10,
    )

    try:
        descriptions = all_narratives.get(narrative_sheet_name)
        if not descriptions:
            print(f"No narratives found for sheet '{narrative_sheet_name}'. Skipping table '{table_key}'.")
            return

        narratives = []
        narrative_keys = []
        for description_key, description_text in descriptions.items():
            if not isinstance(description_text, str) or not description_text:
                print(f"Warning: Invalid or empty description for '{description_key}' in sheet '{narrative_sheet_name}'. Skipping.")
                continue
            narrative_embedding = get_embedding(description_text)
            narrative_embedding_array = np.array(narrative_embedding).reshape(1, -1)
            narratives.append(narrative_embedding_array)
            narrative_keys.append(description_key)

        print(f"Found {len(narratives)} narratives to process for {table_key}.")

        if not narratives:
            print(f"No valid narratives found for table '{table_key}'. Skipping.")
            return

        narratives_matrix = np.vstack(narratives)
        await add_similarity_columns_if_not_exist(pool, source_table, narrative_keys)

        # Build WHERE clause to check if any similarity column is NULL
        similarity_columns = [f"{sanitize_key(key)}_similarity" for key in narrative_keys]
        null_check = ' OR '.join([f"{col} IS NULL" for col in similarity_columns])
        where_clause = f"WHERE {null_check}" if null_check else ""

        set_clauses = [f"{sanitize_key(key)}_similarity = ${i+2}" for i, key in enumerate(narrative_keys)]
        update_query = f"""
            UPDATE {source_table}
            SET {', '.join(set_clauses)}
            WHERE id = $1
        """

        print(f"Fetching messages from {source_table}...")
        async with pool.acquire() as conn:
            total_messages = await conn.fetchval(
                f"SELECT COUNT(*) FROM {source_table} {where_clause} AND {embedding_column} IS NOT NULL"
            )
            print(f"Found {total_messages} messages with missing similarity scores to process in {source_table}.")

            if total_messages == 0:
                print(f"No messages to process in {source_table}.")
                return

            last_id = 0
            processed = 0
            with tqdm(total=total_messages, desc=f"Classifying messages in {source_table}") as pbar:
                while True:
                    query = f"""
                        SELECT id, {embedding_column}
                        FROM {source_table}
                        {where_clause} AND {embedding_column} IS NOT NULL AND id > $1
                        ORDER BY id
                        LIMIT {batch_size}
                    """
                    rows = await conn.fetch(query, last_id)
                    if not rows:
                        break
                    records_to_update = []
                    for row in rows:
                        message_embedding_json = row[embedding_column]
                        if message_embedding_json is None:
                            continue
                        try:
                            if isinstance(message_embedding_json, str):
                                message_embedding = json.loads(message_embedding_json)
                            elif isinstance(message_embedding_json, (list, tuple)):
                                message_embedding = message_embedding_json
                            else:
                                logging.error(f"Unexpected embedding format in row ID {row.get('id', 'N/A')}. Skipping.")
                                continue
                        except json.JSONDecodeError:
                            logging.error(f"Invalid JSON embedding in row ID {row.get('id', 'N/A')}. Skipping.")
                            continue
                        message_embedding_array = np.array(message_embedding).reshape(1, -1)
                        similarities = cosine_similarity(narratives_matrix, message_embedding_array).flatten()
                        params = [row['id']] + [float(sim) for sim in similarities]
                        records_to_update.append(params)
                    if records_to_update:
                        try:
                            async with conn.transaction():
                                await conn.executemany(update_query, records_to_update)
                            pbar.update(len(records_to_update))
                        except Exception as e:
                            logging.error(f"Error bulk updating rows for table {source_table}: {e}")
                            print(f"Error bulk updating rows for table {source_table}: {e}")
                            continue
                    last_id = rows[-1]['id']
    finally:
        await pool.close()
        print(f"Processing for '{table_key}' complete. Connection pool closed.")

async def main():
    """Main function to run the classification process."""
    print("--- Starting narrative classification process ---")
    all_narratives = load_narratives_from_excel(NARRATIVES_FILE_PATH)
    if not all_narratives:
        print("Could not load narratives. Exiting.")
        return

    for table_key, narrative_sheet_name in TABLE_CONFIG.items():
        await process_table(table_key, narrative_sheet_name, all_narratives, batch_size=BATCH_SIZE)
    
    print("\n--- All processing tasks completed ---")

if __name__ == "__main__":
    asyncio.run(main()) 