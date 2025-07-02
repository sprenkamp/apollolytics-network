import os
import psycopg2
from dotenv import load_dotenv

# Set DRY_RUN to False to actually move rows
DRY_RUN = False

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

COLUMN = "messagetext"
MIN_LENGTH = 250
DATE_COLUMN = "messagedate"
HIGHEST_DATE = "2022-01-01"  # ISO format

conn = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)
cur = conn.cursor()

cur.execute("""
    SELECT tablename FROM pg_tables WHERE schemaname = 'public';
""")
tables = [row[0] for row in cur.fetchall()]

for table in tables:
    removed_table = f"{table}_removed"
    # Check if the columns exist in this table
    cur.execute("""
        SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name = %s;
    """, (table, COLUMN))
    has_text_col = bool(cur.fetchone())
    cur.execute("""
        SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name = %s;
    """, (table, DATE_COLUMN))
    has_date_col = bool(cur.fetchone())
    if not has_text_col and not has_date_col:
        continue
    # Build where clause for rows to be REMOVED
    where_clauses = []
    params = []
    if has_text_col:
        where_clauses.append(f"LENGTH({COLUMN}) < %s")
        params.append(MIN_LENGTH)
    if has_date_col:
        where_clauses.append(f"{DATE_COLUMN} < %s")
        params.append(HIGHEST_DATE)
    where_sql = " OR ".join(where_clauses)
    # Count rows to be removed
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_sql}", tuple(params))
    to_remove = cur.fetchone()[0]
    # Count rows to be kept
    cur.execute(f"SELECT COUNT(*) FROM {table}", ())
    total_rows = cur.fetchone()[0]
    to_keep = total_rows - to_remove
    if to_keep == 0 or total_rows == 0:
        continue
    # Estimate average row size in the original table
    cur.execute(f"SELECT pg_total_relation_size(%s)", (table,))
    total_size_bytes = cur.fetchone()[0]
    avg_row_size = total_size_bytes / total_rows if total_rows else 0
    estimated_keep_size_mb = (avg_row_size * to_keep) / (1024 * 1024)
    print(f"{table}: {estimated_keep_size_mb:.2f} MB would remain ({to_keep} rows) [chars >= {MIN_LENGTH} and {DATE_COLUMN} >= {HIGHEST_DATE}]")
    if DRY_RUN:
        continue
    # Actually move rows if not in DRY_RUN
    # Create the _removed table if it doesn't exist
    cur.execute(f"CREATE TABLE IF NOT EXISTS {removed_table} (LIKE {table} INCLUDING ALL);")
    # Move rows to the _removed table
    cur.execute(f"INSERT INTO {removed_table} SELECT * FROM {table} WHERE {where_sql}", tuple(params))
    moved = cur.rowcount
    # Delete the moved rows from the original table
    cur.execute(f"DELETE FROM {table} WHERE {where_sql}", tuple(params))
    deleted = cur.rowcount
    conn.commit()
    print(f"{table}: Moved and deleted {deleted} rows to {removed_table}.")

cur.close()
conn.close()