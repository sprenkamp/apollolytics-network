import os
import asyncio
import logging
import asyncpg
from dotenv import load_dotenv
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)  # Adjust logging level as needed

# PostgreSQL connection details from environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "kiliansprenkamp")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # Empty if not set

# Check if MPS is available and set the device accordingly
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("MPS device found. Using MPS for computations.")
else:
    device = torch.device("cpu")
    logging.info("MPS device not found. Using CPU for computations.")

# Load the embedding model on the specified device
embedding_model_name = "deepvk/USER-bge-m3"
model = SentenceTransformer(embedding_model_name, device=device)

# Adjustable batch size
BATCH_SIZE = 1000  # Adjusted for better performance with MPS

async def ensure_column_exists(connection, column_name):
    """
    Ensure that a specific column exists in the database. Create it if it doesn't exist.
    """
    # Use LOWER() to ensure case-insensitive comparison
    column_check_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'messages_community' AND LOWER(column_name) = LOWER('{column_name}')
    """
    result = await connection.fetch(column_check_query)
    if not result:
        # Only attempt to add the column if it does not exist
        await connection.execute(f'ALTER TABLE messages_community ADD COLUMN "{column_name}" JSONB DEFAULT NULL;')
        logging.info(f"Added column: {column_name}")
    else:
        logging.info(f"Column {column_name} already exists.")

def chunk_list(lst, chunk_size):
    """Yield successive chunks of a specified size from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

async def add_embeddings():
    """Connect to the database and add embeddings for messageText."""
    # Ensure column_name is in lowercase to match PostgreSQL behavior
    column_name = f"embedding_{embedding_model_name.split('/')[-1].replace('-', '_').lower()}"

    try:
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
            # Ensure the embedding column exists
            await ensure_column_exists(connection, column_name)

            # Fetch messages outside of a transaction
            rows = await connection.fetch(f"""
                SELECT chat_id, id, messageText
                FROM messages_community
                WHERE "{column_name}" IS NULL AND messageText IS NOT NULL
            """)

            total_messages = len(rows)
            logging.info(f"Found {total_messages} messages to process.")

            # Break the rows into batches
            batches = list(chunk_list(rows, BATCH_SIZE))
            total_batches = len(batches)
            logging.info(f"Processing messages in {total_batches} batches of up to {BATCH_SIZE} messages each.")

            # Initialize a single tqdm progress bar
            with tqdm(total=total_messages, desc='Processing messages') as progress_bar:
                for batch_num, batch in enumerate(batches, start=1):
                    try:
                        # Start a transaction for each batch
                        async with connection.transaction():
                            for row in batch:
                                try:
                                    # Generate embedding using the SentenceTransformer model
                                    # Move input text to device if necessary
                                    embedding = model.encode(
                                        row["messagetext"],
                                        convert_to_tensor=True,
                                        device=device,
                                        show_progress_bar=False  # Suppress per-batch progress bar
                                    ).cpu().tolist()

                                    # Convert embedding to JSON string
                                    embedding_json = json.dumps(embedding)

                                    # Update the database
                                    await connection.execute(
                                        f"""
                                        UPDATE messages_community
                                        SET "{column_name}" = $1::jsonb
                                        WHERE chat_id = $2 AND id = $3
                                        """,
                                        embedding_json,
                                        row["chat_id"],
                                        row["id"],
                                    )

                                except Exception as e:
                                    logging.error(f"Error processing message {row['id']}: {e}")
                                finally:
                                    # Update the progress bar
                                    progress_bar.update(1)
                        # The transaction for this batch will auto-commit here
                        logging.info(f"Batch {batch_num}/{total_batches} committed successfully.")
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_num}: {e}")
        await db_pool.close()
    except Exception as e:
        logging.error(f"Database connection error: {e}")

if __name__ == "__main__":
    asyncio.run(add_embeddings())
