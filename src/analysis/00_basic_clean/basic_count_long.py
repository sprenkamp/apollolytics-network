import os
import psycopg2
from dotenv import load_dotenv

# Always DRY_RUN: only count, never modify
DRY_RUN = True

# Load environment variables from .env file
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

COLUMN = "messagetext"  # Text column to check
MIN_WORDS = 350           # Minimum number of words
MIN_CHARS = 2000          # Minimum number of characters

print(f"Counting rows with more than {MIN_WORDS} words and {MIN_CHARS} characters in column '{COLUMN}'.")

conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)
cur = conn.cursor()

# Get all user tables
cur.execute("""
    SELECT tablename FROM pg_tables WHERE schemaname = 'public';
""")
tables = [row[0] for row in cur.fetchall()]

for table in tables:
    # Check if the column exists in this table
    cur.execute("""
        SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name = %s;
    """, (table, COLUMN))
    has_text_col = bool(cur.fetchone())
    if not has_text_col:
        print(f"Skipping {table}: '{COLUMN}' not found.")
        continue
    # Count how many rows match the criteria
    cur.execute(
        f"""
        SELECT COUNT(*) FROM {table}
        WHERE LENGTH({COLUMN}) > %s AND array_length(regexp_split_to_array({COLUMN}, '\\s+'), 1) > %s
        """,
        (MIN_CHARS, MIN_WORDS)
    )
    count = cur.fetchone()[0]
    print(f"{table}: {count} rows have more than {MIN_WORDS} words and {MIN_CHARS} characters.")

cur.close()
conn.close() 