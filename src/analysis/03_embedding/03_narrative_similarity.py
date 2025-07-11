import os
import math
import json
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "your_username")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

TABLES = [
    "russian_channels_messages",
    "russian_groups_messages",
    "ukrainian_channels_messages",
    "ukrainian_groups_messages"
]

TARGET_TABLE = "relevant_classified_narrative_results"

# Connect to DB
engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def get_columns_for_table(table, exclude=None):
    query = f"SELECT * FROM {table} LIMIT 0"
    df = pd.read_sql(query, engine)
    columns = df.columns.tolist()
    if exclude and exclude in columns:
        columns.remove(exclude)
    return columns

def get_narrative_similarity_columns(columns):
    # Adjust this logic if your similarity columns have a different naming pattern
    return [col for col in columns if col.lower().endswith('_similarity')]

def safe_stringify(val):
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return val

# Get the union of all columns across all tables (excluding 'multilingual_e5_large_instruct')
def get_all_columns():
    all_cols = set()
    for table in TABLES:
        cols = get_columns_for_table(table, exclude='multilingual_e5_large_instruct')
        all_cols.update(cols)
    # Add extra columns
    all_cols.update(['country', 'type', 'table'])
    return list(all_cols)

ALL_COLUMNS = get_all_columns()

def process_table_in_batches(table, batch_size=250000, first_batch=False):
    columns = get_columns_for_table(table, exclude='multilingual_e5_large_instruct')
    narrative_sim_cols = get_narrative_similarity_columns(columns)
    # Get total number of rows for progress bar
    count_query = f"SELECT COUNT(*) FROM {table}"
    total_rows = pd.read_sql(count_query, engine).iloc[0, 0]
    num_batches = math.ceil(total_rows / batch_size)
    query = f"SELECT {', '.join(columns)} FROM {table}"
    chunks = pd.read_sql(query, engine, chunksize=batch_size)
    for i, chunk in enumerate(tqdm(chunks, desc=f"Processing batches for {table}", total=num_batches)):
        # Add country column
        if "russian" in table:
            chunk['country'] = "Russia"
        elif "ukrainian" in table:
            chunk['country'] = "Ukraine"
        else:
            chunk['country'] = "Unknown"
        # Add type column (channel/group)
        if "channels" in table:
            chunk['type'] = "channel"
        elif "groups" in table:
            chunk['type'] = "group"
        else:
            chunk['type'] = "unknown"
        chunk['table'] = table
        # Reorder columns: country, type, table, then the rest
        first_cols = ['country', 'type', 'table']
        rest_cols = [col for col in chunk.columns if col not in first_cols]
        chunk = chunk[first_cols + rest_cols]
        # Filter: keep only rows where any narrative similarity > 0.85
        if narrative_sim_cols:
            mask = (chunk[narrative_sim_cols] > 0.85).any(axis=1)
            filtered_chunk = chunk[mask]
        else:
            filtered_chunk = pd.DataFrame(columns=chunk.columns)  # No narrative columns, skip
        # Convert dict or list columns to JSON strings before writing to SQL
        for col in filtered_chunk.columns:
            if filtered_chunk[col].apply(lambda x: isinstance(x, (dict, list))).any():
                filtered_chunk[col] = filtered_chunk[col].apply(safe_stringify)
        # Reindex to ALL_COLUMNS to ensure schema consistency
        filtered_chunk = filtered_chunk.reindex(columns=ALL_COLUMNS)
        # Reset index to avoid column suffixes in SQL
        filtered_chunk = filtered_chunk.reset_index(drop=True)
        # Write to Postgres if there are rows
        if not filtered_chunk.empty:
            if first_batch and i == 0:
                filtered_chunk.to_sql(TARGET_TABLE, engine, if_exists='replace', index=False)
            else:
                filtered_chunk.to_sql(TARGET_TABLE, engine, if_exists='append', index=False)
            print(f"Saved filtered batch {i+1} from {table} ({len(filtered_chunk)} rows) to {TARGET_TABLE}")
        else:
            print(f"No relevant rows in batch {i+1} from {table}")

def main():
    first_batch = True
    for table in TABLES:
        process_table_in_batches(table, batch_size=250000, first_batch=first_batch)
        first_batch = False  # Only replace on very first batch

if __name__ == "__main__":
    main() 