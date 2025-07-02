import os
import argparse
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """)
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    return tables

def describe_table(conn, table_name):
    print(f"\n--- Descriptive Analysis for table: {table_name} ---\n")
    # Get column info
    cur = conn.cursor()
    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (table_name,))
    columns = cur.fetchall()
    # print("Columns:")
    # for col, dtype in columns:
    #     print(f"  {col}: {dtype}")
    # print()
    # Get row count
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cur.fetchone()[0]
    print(f"Row count: {row_count}\n")
    # Load the entire table into pandas
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    if df.empty:
        print(f"Table {table_name} is empty. Skipping analysis.\n")
        cur.close()
        return None, None
    # Nulls per column
    # print("Nulls per column:")
    # # print(df.isnull().sum())
    # print()
    # Basic stats for numeric/date columns
    print("Basic statistics (numeric/date columns):")
    print(df.describe(include=["number", "datetime"]))
    print()
    # Prepare for combined plots
    messages_per_day_df = None
    channel_counts = None
    if 'messagedate' in df.columns:
        df['messagedate'] = pd.to_datetime(df['messagedate'], errors='coerce')
        messages_per_day = df.groupby(df['messagedate'].dt.date).size()
        messages_per_day_df = messages_per_day.reset_index(name='count')
        messages_per_day_df.rename(columns={'messagedate': 'date'}, inplace=True)
        messages_per_day_df['table'] = table_name
    if 'chat_name' in df.columns:
        channel_counts = df['chat_name'].value_counts().head(10)
    # Top 5 most liked/shared messages (by views, forwards, reactions)
    # Views
    cols = [col for col in ['id', 'chat_name', 'messagetext', 'views'] if col in df.columns]
    if 'views' in df.columns and len(cols) >= 2:
        print("\nTop 5 messages by views:")
        for _, row in df[cols].sort_values('views', ascending=False).head(5).iterrows():
            print(f"views: {row.get('views', '')}\nmessageText: {row.get('messagetext', '')}\n---")
    elif 'views' in df.columns:
        print("\nNot enough columns to display top messages by views.")
    # Forwards
    cols = [col for col in ['id', 'chat_name', 'messagetext', 'forwards'] if col in df.columns]
    if 'forwards' in df.columns and len(cols) >= 2:
        print("\nTop 5 messages by forwards:")
        for _, row in df[cols].sort_values('forwards', ascending=False).head(5).iterrows():
            print(f"forwards: {row.get('forwards', '')}\nmessageText: {row.get('messagetext', '')}\n---")
    elif 'forwards' in df.columns:
        print("\nNot enough columns to display top messages by forwards.")
    # Reactions
    if 'reactions' in df.columns:
        print("\nTop 5 messages by reactions:")
        def reaction_count(val):
            if pd.isnull(val):
                return 0
            try:
                import json
                d = json.loads(val) if isinstance(val, str) else val
                return sum(d.values()) if isinstance(d, dict) else 0
            except Exception:
                return 0
        df['reaction_count'] = df['reactions'].apply(reaction_count)
        cols = [col for col in ['id', 'chat_name', 'messagetext', 'reaction_count', 'reactions'] if col in df.columns]
        if len(cols) >= 2:
            for _, row in df[cols].sort_values('reaction_count', ascending=False).head(5).iterrows():
                print(f"Total reactions: {row.get('reaction_count', '')}\nMessage text: {row.get('messagetext', '')}\nReactions breakdown: {row.get('reactions', '')}\n---")
        else:
            print("Not enough columns to display top messages by reactions.")
    print()
    cur.close()
    return messages_per_day_df, channel_counts

def plot_combined_time_series(messages_per_day_dict):
    # Concatenate all non-empty DataFrames
    dfs = [df for df in messages_per_day_dict.values() if df is not None and not df.empty]
    if not dfs:
        print("No data available for time series plot. Skipping plot.")
        return
    all_df = pd.concat(dfs, ignore_index=True)
    plt.figure(figsize=(12, 7))
    for table in all_df['table'].unique():
        sub = all_df[all_df['table'] == table]
        plt.plot(sub['date'], sub['count'], label=table)
    plt.xlabel('Date')
    plt.ylabel('Number of messages')
    plt.title('Messages over time (all tables)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_tables_messages_over_time.png'))
    plt.close()
    print(f"Saved combined time series plot: all_tables_messages_over_time.png")

def plot_combined_pies(channel_counts_dict):
    n = len(channel_counts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    for ax, (table, counts) in zip(axes, channel_counts_dict.items()):
        if counts is not None and not counts.empty:
            counts.plot.pie(ax=ax, autopct='%1.1f%%', legend=False)
            ax.set_title(f"{table}")
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_tables_channels_pies.png'))
    plt.close()
    print(f"Saved combined pie chart: all_tables_channels_pies.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descriptive analysis of all PostgreSQL tables.")
    # No limit argument
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    tables = get_all_tables(conn)
    print(f"Found tables: {tables}")
    messages_per_day_dict = {}
    channel_counts_dict = {}
    for table in tables:
        messages_per_day_df, channel_counts = describe_table(conn, table)
        messages_per_day_dict[table] = messages_per_day_df
        channel_counts_dict[table] = channel_counts
    # Plot combined charts
    print(messages_per_day_dict)
    plot_combined_time_series(messages_per_day_dict)
    plot_combined_pies(channel_counts_dict)
    conn.close() 