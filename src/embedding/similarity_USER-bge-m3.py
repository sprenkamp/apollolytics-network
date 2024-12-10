import os
import asyncio
import asyncpg
import numpy as np
import json
import csv
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity  # Importing cosine_similarity
from sentence_transformers import SentenceTransformer  # Added SentenceTransformer import
import torch  # Import torch here since we use it for device checking

# Load environment variables
load_dotenv()

# PostgreSQL connection details from environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "your_username")  # Replace with your username
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # Empty if not set

# Initialize SentenceTransformer model
EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
# You can set device to 'cuda' if you have a GPU, otherwise 'cpu'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def get_embedding(text):
    """Generate embedding for the given text using SentenceTransformer."""
    return MODEL.encode(text).tolist()

async def fetch_embeddings():
    """Fetch messages with embeddings from the database."""
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

    async with db_pool.acquire() as connection:
        # Fetch column names dynamically
        # column_query = f"""
        #     SELECT column_name
        #     FROM information_schema.columns
        #     WHERE table_name = 'messages'
        # """
        # columns = await connection.fetch(column_query)
        # column_names = [record['column_name'] for record in columns]

        # # Fetch messages with embeddings
        # select_columns = ', '.join(column_names)
        rows = await connection.fetch(f"""
            SELECT {select_columns}
            FROM messages
            WHERE {column_name} IS NOT NULL
        """)
    await db_pool.close()
    return rows, column_names

async def process_descriptions(descriptions, output_csv, threshold=0.8):
    """Process descriptions and save results in a CSV."""
    rows, column_names = await fetch_embeddings()
    column_name = f"embedding_{EMBEDDING_MODEL_NAME.split('/')[-1].replace('-', '_').lower()}"

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header with all column names plus additional fields
        header = ["Description", "Similarity"] + column_names
        csv_writer.writerow(header)

        for description_key, description_dict in descriptions.items():
            # Use the Russian description
            description_text = description_dict.get('ru', '')
            if not description_text:
                print(f"Warning: No Russian description for {description_key}. Skipping.")
                continue
            print(f"Processing description: {description_key}")
            description_embedding = get_embedding(description_text)
            description_embedding_array = np.array(description_embedding).reshape(1, -1)

            similar_messages = []

            for row in tqdm(rows, desc=f"Calculating similarities for {description_key}"):
                message_embedding_json = row[column_name]
                if message_embedding_json is None:
                    continue

                # Assuming the embedding is stored as a JSON array
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
                # print(message_embedding_array, description_embedding_array)
                # Compute cosine similarity using sklearn
                similarity = cosine_similarity(description_embedding_array, message_embedding_array)[0][0]
                # print(similarity)
                if similarity >= threshold:
                    similar_messages.append((similarity, row))

            for similarity, row in similar_messages:
                row_data = [description_key, similarity]
                # Extract all column data in order
                for col_name in column_names:
                    value = row[col_name]
                    # Handle JSON fields if any
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    row_data.append(value)
                csv_writer.writerow(row_data)
    print(f"Results saved to {output_csv}")

async def main():
    # Load descriptions from JSON file
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
    THRESHOLD = 0.5  # Adjust based on your needs

    await process_descriptions(descriptions, OUTPUT_CSV, THRESHOLD)

if __name__ == "__main__":
    asyncio.run(main())
