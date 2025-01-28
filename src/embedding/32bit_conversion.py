import os
import asyncio
import asyncpg
import numpy as np
import json
import csv
from decimal import *
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import math  # New import for mathematical operations

load_dotenv()

# Configuration Variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "your_username")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

# New Variable to Choose the Table
TABLE_NAME = os.getenv("TABLE_NAME", "messages_community")  # Set to "messages_news" or "messages_community"

async def ensure_column_exists(db_pool, column_name):
    """
    Ensure that a specific column exists in the specified table. Create it if it doesn't exist.
    """
    # Use LOWER() to ensure case-insensitive comparison
    column_check_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = '{TABLE_NAME}' AND LOWER(column_name) = LOWER('{column_name}')
    """
    async with db_pool.acquire() as connection:
        result = await connection.fetch(column_check_query)
        if not result:
            # Only attempt to add the column if it does not exist
            await connection.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{column_name}" JSONB DEFAULT NULL;')
            print(f"Added column: {column_name} to table {TABLE_NAME}")
        else:
            print(f"Column {column_name} already exists in table {TABLE_NAME}.")

async def fetch_embeddings(db_pool, column_name, fetch_columns, batch_size=1000, nb_batches=None):
    offset = 0
    while nb_batches is None or nb_batches > 0:
        query = f"""
            SELECT {", ".join(fetch_columns)}
            FROM {TABLE_NAME}
            WHERE {column_name} IS NOT NULL
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """
        async with db_pool.acquire() as connection:
            rows = await connection.fetch(query)
        if not rows:
            break
        yield rows
        offset += batch_size
        if nb_batches:
            nb_batches -= 1

async def process_descriptions(selected_read_columns, batch_size=1000, nb_batches=None):
    embedding_suffix = EMBEDDING_MODEL_NAME.split('/')[-1].replace('-', '_').lower()
    column_name = f"embedding_{embedding_suffix}"
    
    db_pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        min_size=1,
        max_size=10,
    )
    await ensure_column_exists(db_pool, column_name + '_32_bits')

    # Step 1: Count total rows to process
    async with db_pool.acquire() as connection:
        count_query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {column_name} IS NOT NULL"
        total_rows = await connection.fetchval(count_query)
    
    if total_rows == 0:
        print(f"No rows to process in table {TABLE_NAME}.")
        await db_pool.close()
        return

    # Step 2: Calculate total number of batches
    total_batches = math.ceil(total_rows / batch_size)
    if nb_batches:
        total_batches = min(total_batches, nb_batches)

    print(f"Total rows to process: {total_rows}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches to process: {total_batches}")

    current_batch = 0  # Initialize current batch counter

    async for rows in fetch_embeddings(db_pool, column_name, selected_read_columns, batch_size=batch_size, nb_batches=nb_batches):
        current_batch += 1
        print(f"Processing batch {current_batch}/{total_batches}")
        getcontext().prec = 8
        for row in tqdm(rows, leave=False):
            message_embedding_json = row[column_name]
            if message_embedding_json is None:
                continue
            if isinstance(message_embedding_json, str):
                try:
                    message_embedding = json.loads(message_embedding_json)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON embedding in row {row.get('id', 'N/A')}. Skipping.")
                    continue
            elif isinstance(message_embedding_json, (list, tuple)):
                message_embedding = message_embedding_json
            else:
                print(f"Warning: Unexpected embedding format in row {row.get('id', 'N/A')}. Skipping.")
                continue
            message_embedding_array = np.array(message_embedding).reshape(1, -1)
            embedding_json = json.dumps([[float(Decimal("%.8f" % v)) for v in  message_embedding_array.tolist()[0]]])
            async with db_pool.acquire() as connection:
                async with connection.transaction():
                    # Update the database
                    await connection.execute(
                        f"""
                        UPDATE {TABLE_NAME}
                        SET "{column_name}_32_bits" = $1::jsonb
                        WHERE chat_id = $2 AND id = $3
                        """,
                        embedding_json,
                        row["chat_id"],
                        row["id"],
                    )

    await db_pool.close()

async def main():
    # Specify which columns to read from the database (must include the embedding column)
    embedding_suffix = EMBEDDING_MODEL_NAME.split('/')[-1].replace('-', '_').lower()
    embedding_column = f"embedding_{embedding_suffix}"
    selected_read_columns = ["id", "messagetext", "chat_id", "messagedatetime", embedding_column]
    # Specify which columns to write to the CSV (if needed)
    selected_write_columns = ["id", "messagetext", "chat_id", "messagedatetime"]

    # Set nb_batches to None to process all batches, or set to a specific number to limit
    await process_descriptions(selected_read_columns, batch_size=100_000, nb_batches=None)

if __name__ == "__main__":
    asyncio.run(main())
