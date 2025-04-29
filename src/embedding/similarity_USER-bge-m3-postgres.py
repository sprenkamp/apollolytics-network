import os
import asyncio
import asyncpg
import numpy as np
import json
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import logging
import re

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

EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

SIMILARITY_THRESHOLD = 0.6
BATCH_SIZE = 250000
TYPES = ['news', 'community']
DESCRIPTIONS_FILE = 'src/embedding/narratives/narratives.json'

# ==========================
# Helper Functions
# ==========================

def sanitize_key(key):
    sanitized = re.sub(r'[ \-/]', '_', key)
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)
    return sanitized

def get_embedding(text):
    return MODEL.encode(text).tolist()

async def create_table_if_not_exists(target_pool, table_name, base_columns, narrative_keys):
    """
    Create a new table in the target database if it doesn't exist with correct data types,
    including similarity columns for each narrative.
    A surrogate key 'record_id' is used as the primary key.
    """
    base_column_type_mapping = {
        "id": "BIGINT",
        "messagetext": "TEXT",
        "chat_id": "BIGINT",
        "chat_name": "TEXT",
        "messagedatetime": "TIMESTAMP",
    }

    column_definitions = []
    # Surrogate primary key
    column_definitions.append("record_id SERIAL PRIMARY KEY")

    for col in base_columns:
        if col in base_column_type_mapping:
            column_definitions.append(f"{col} {base_column_type_mapping[col]}")
        else:
            column_definitions.append(f"{col} TEXT")

    for key in narrative_keys:
        sanitized_key = sanitize_key(key)
        column_definitions.append(f"{sanitized_key}_similarity FLOAT")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(column_definitions)}
    )
    """
    try:
        async with target_pool.acquire() as connection:
            await connection.execute(create_table_query)
        print(f"Table '{table_name}' is ready in the database '{POSTGRES_DB}'.")
    except Exception as e:
        logging.error(f"Failed to create table '{table_name}': {e}")
        print(f"Failed to create table '{table_name}': {e}")

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

async def process_type(type_name, descriptions, batch_size=1000):
    source_table = f"messages_{type_name}"
    target_table = f"classified_messages_{type_name}"
    embedding_column = "embedding_user_bge_m3_32_bits"
    selected_read_columns = ["id", "messagetext", "chat_id", "chat_name", "messagedatetime", embedding_column]
    base_write_columns = ["id", "messagetext", "chat_id", "chat_name", "messagedatetime"]

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
        narratives = []
        narrative_keys = []
        for description_key, description_dict in descriptions.items():
            description_text = description_dict.get('ru', '')
            if not description_text:
                print(f"Warning: No Russian description for '{description_key}'. Skipping.")
                continue
            narrative_embedding = get_embedding(description_text)
            narrative_embedding_array = np.array(narrative_embedding).reshape(1, -1)
            narratives.append(narrative_embedding_array)
            narrative_keys.append(description_key)

        if not narratives:
            print(f"No valid narratives found for type '{type_name}'. Skipping.")
            return

        narratives_matrix = np.vstack(narratives)
        await create_table_if_not_exists(pool, target_table, base_write_columns, narrative_keys)

        async with pool.acquire() as conn:
            async for rows in fetch_embeddings_keyset(
                conn,
                source_table,
                embedding_column,
                selected_read_columns,
                batch_size=batch_size
            ):
                for row in tqdm(rows, desc=f"Processing '{type_name}' messages", leave=False):
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
                    row_data = []
                    for col in base_write_columns:
                        value = row[col]
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, ensure_ascii=False)
                        row_data.append(value)

                    for sim in similarities:
                        row_data.append(float(sim))
                    
                    placeholders = ','.join([f'${i}' for i in range(1, len(row_data) + 1)])
                    similarity_columns = [f"{sanitize_key(key)}_similarity" for key in narrative_keys]
                    all_columns = base_write_columns + similarity_columns
                    insert_query = f"""
                        INSERT INTO {target_table} ({", ".join(all_columns)})
                        VALUES ({placeholders})
                    """
                    try:
                        async with conn.transaction():
                            await conn.execute(insert_query, *row_data)
                    except Exception as e:
                        logging.error(f"Error inserting row with data {row_data}: {e}")
                        print(f"Error inserting row with data {row_data}: {e}")
                        continue

        print(f"Processing for type '{type_name}' completed and stored in table '{target_table}'.")
    except Exception as e:
        logging.error(f"Unexpected error in process_type for '{type_name}': {e}")
        print(f"Unexpected error in process_type for '{type_name}': {e}")
    finally:
        await pool.close()

async def main():
    try:
        with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
    except FileNotFoundError:
        logging.error(f"'{DESCRIPTIONS_FILE}' not found.")
        print(f"Error: '{DESCRIPTIONS_FILE}' not found.")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{DESCRIPTIONS_FILE}': {e}")
        print(f"Error decoding JSON from '{DESCRIPTIONS_FILE}': {e}")
        return

    tasks = [process_type(type_name, descriptions, batch_size=BATCH_SIZE) for type_name in TYPES]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
