import os
import json
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, Json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# --- CONFIGURATION ---
DRY_RUN = False  # Set to False to process all main tables
ALL_TABLES = [
    "russian_channels_messages",
    "russian_groups_messages",
    "ukrainian_channels_messages",
    "ukrainian_groups_messages"
]
TEST_TABLE = ["test_100k"]
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_COLUMN = "multilingual_e5_large_instruct"
DB_BATCH_SIZE = 100000      # Number of rows to fetch per server-side fetchmany
EMBED_BATCH_SIZE = 128      # Number of texts to encode at once

# --- ENVIRONMENT ---
load_dotenv()
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# --- SETUP DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
logging.info(f"Using device: {device}")

# --- MODEL ---
model = SentenceTransformer(MODEL_NAME, device=device)
if device == "cuda":
    model.half()

# --- DB CONNECT & PROCESS ---
with psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
) as conn:
    tables = TEST_TABLE if DRY_RUN else ALL_TABLES
    # Ensure embedding column exists
    with conn.cursor() as setup_cur:
        for table in tables:
            setup_cur.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{table}' AND column_name = '{EMBEDDING_COLUMN}'
                    ) THEN
                        ALTER TABLE {table} ADD COLUMN {EMBEDDING_COLUMN} JSONB;
                    END IF;
                END$$;
            """)
        conn.commit()

    for table in tables:
        logging.info(f"Processing {table}")
        # Count rows
        with conn.cursor() as cnt_cur:
            cnt_cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cnt_cur.fetchone()[0]
        logging.info(f"Total rows: {total}")

        # iterate via server-side cursor
        with conn.cursor(name=f"stream_{table}") as stream_cur:
            stream_cur.itersize = DB_BATCH_SIZE
            stream_cur.execute(
                f"SELECT id, messagetext FROM {table} WHERE {EMBEDDING_COLUMN} IS NULL ORDER BY id"
            )

            fetched = 0
            batch_num = 0
            while True:
                rows = stream_cur.fetchmany(DB_BATCH_SIZE)
                if not rows:
                    break
                batch_num += 1

                # Skip rows with empty or null messagetext
                filtered = [(r[0], r[1]) for r in rows if r[1] and r[1].strip()]
                if not filtered:
                    fetched += len(rows)
                    logging.info(f"Batch {batch_num}: All rows had empty or null messagetext, skipping.")
                    continue
                ids, texts = zip(*filtered)

                embeddings = []
                with tqdm(total=len(texts), desc=f"{table} batch {batch_num}") as pbar:
                    for start in range(0, len(texts), EMBED_BATCH_SIZE):
                        chunk = texts[start:start+EMBED_BATCH_SIZE]
                        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                            emb = model.encode(
                                chunk,
                                batch_size=EMBED_BATCH_SIZE,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                        embeddings.extend(emb)
                        pbar.update(len(chunk))

                # update in bulk, cast to jsonb in SQL
                emb_json = [Json(e.tolist()) for e in embeddings]
                values = list(zip(ids, emb_json))
                sql = (
                    f"UPDATE {table} AS t SET {EMBEDDING_COLUMN} = v.embedding::jsonb"
                    f" FROM (VALUES %s) AS v(id, embedding) WHERE t.id = v.id"
                )
                with conn.cursor() as upd_cur:
                    execute_values(upd_cur, sql, values, page_size=DB_BATCH_SIZE)
                # DO NOT commit here!
                fetched += len(rows)
                logging.info(f"Updated {fetched} rows in {table}")

        # Commit after the named cursor is closed
        conn.commit()

    logging.info("All done.")
