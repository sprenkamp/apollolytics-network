import os
import asyncio
import asyncpg
import numpy as np
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import math

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "your_username")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def get_embedding(text):
    return MODEL.encode(text).tolist()

async def count_total_rows(db_pool, column_name):
    count_query = f"""
        SELECT COUNT(*) AS total
        FROM messages
        WHERE {column_name} IS NOT NULL
    """
    async with db_pool.acquire() as connection:
        result = await connection.fetchval(count_query)
    return result

async def fetch_embeddings(db_pool, column_name, fetch_columns, batch_size=1000):
    total_rows = await count_total_rows(db_pool, column_name)
    if total_rows == 0:
        return
    total_batches = math.ceil(total_rows / batch_size)
    offset = 0
    batch_num = 1
    while True:
        query = f"""
            SELECT {", ".join(fetch_columns)}
            FROM messages
            WHERE {column_name} IS NOT NULL
            ORDER BY id
            LIMIT {batch_size} OFFSET {offset}
        """
        async with db_pool.acquire() as connection:
            rows = await connection.fetch(query)
        if not rows:
            break
        print(f"Processing batch {batch_num}/{total_batches}...")
        yield rows
        offset += batch_size
        batch_num += 1

async def process_descriptions(descriptions, output_csv, selected_read_columns, selected_write_columns, threshold=0.8, batch_size=1000):
    column_name = f"embedding_{EMBEDDING_MODEL_NAME.split('/')[-1].replace('-', '_').lower()}"
    db_pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        min_size=1,
        max_size=10,
    )

    # Precompute narrative embeddings
    narratives = []
    for description_key, description_dict in descriptions.items():
        description_text = description_dict.get('ru', '')
        if not description_text:
            print(f"Warning: No Russian description for {description_key}. Skipping.")
            continue
        narrative_embedding = get_embedding(description_text)
        narrative_embedding_array = np.array(narrative_embedding).reshape(1, -1)
        narratives.append((description_key, narrative_embedding_array))

    # Prepare header for CSV: includes one column per narrative similarity
    narrative_columns = [f"{desc_key}_similarity" for desc_key, _ in narratives]
    header = selected_write_columns + narrative_columns

    # Open CSV for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

        # Outer loop: Fetch DB rows in batches
        async for rows in fetch_embeddings(db_pool, column_name, selected_read_columns, batch_size=batch_size):
            # Inner loop: For each message in this batch, compute similarity with all narratives
            for row in tqdm(rows, desc=f"Calculating similarities for batch", leave=False):
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

                # Compute similarities for all narratives
                similarity_values = []
                for description_key, narrative_embedding_array in narratives:
                    similarity = cosine_similarity(narrative_embedding_array, message_embedding_array)[0][0]
                    # If you want to filter by threshold:
                    # similarity = similarity if similarity >= threshold else 0
                    similarity_values.append(similarity)

                # Write single row with all similarities as columns
                row_data = []
                for col_name in selected_write_columns:
                    value = row[col_name]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    row_data.append(value)
                row_data.extend(similarity_values)

                csv_writer.writerow(row_data)

    await db_pool.close()
    print(f"Results saved to {output_csv}")

async def main():
    DESCRIPTIONS_FILE = 'src/embedding/narratives/narratives.json'
    try:
        with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {DESCRIPTIONS_FILE} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {DESCRIPTIONS_FILE}: {e}")
        return

    OUTPUT_CSV = "similar_messages_BGE.csv"
    THRESHOLD = 0.0

    selected_read_columns = ["id", "messagetext", "chat_id", "chat_name", 'messagedatetime', f"embedding_{EMBEDDING_MODEL_NAME.split('/')[-1].replace('-', '_').lower()}"]
    selected_write_columns = ["id", "messagetext", "chat_id", "chat_name", 'messagedatetime']

    await process_descriptions(descriptions, OUTPUT_CSV, selected_read_columns, selected_write_columns, THRESHOLD, batch_size=200000)

if __name__ == "__main__":
    asyncio.run(main())
