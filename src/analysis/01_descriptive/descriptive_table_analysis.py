import os
import argparse
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from io import StringIO
import gc
import psycopg2.extras
from sqlalchemy import create_engine, text

# Load environment variables from .env file
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Target tables for analysis (only base tables, removed versions will be merged)
TARGET_TABLES = [
    "russian_channels_messages",
    "russian_groups_messages", 
    "ukrainian_channels_messages",
    "ukrainian_groups_messages"
]

# Columns to include in analysis
SELECTED_COLUMNS = [
    'chat_id', 'id', 'chat_name', 'peer_id',
    'messagedatetime', 'messagedate', 'messagetext', 'out', 'mentioned',
    'media_unread', 'silent', 'post', 'from_scheduled', 'legacy',
    'edit_hide', 'pinned', 'noforwards', 'invert_media', 'offline',
    'from_id', 'from_boosts_applied', 'saved_peer_id', 'fwd_from',
    'fwd_from_type', 'via_bot_id', 'via_business_bot_id', 'reply_to',
    'reply_markup', 'entities', 'edit_date', 'post_author', 'grouped_id',
    'ttl_period', 'quick_reply_shortcut_id', 'effect', 'factcheck', 'views',
    'forwards', 'replies', 'reactions'
]

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Descriptive analysis of all PostgreSQL tables.")
parser.add_argument("--chunk-size", type=int, default=10000, 
                   help="Number of rows to process at once (default: 10000)")
args = parser.parse_args()

# Chunk size for processing large tables
CHUNK_SIZE = args.chunk_size

def get_all_tables(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """)
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    return tables

def get_available_columns(conn, table_name):
    """Get available columns for a table"""
    cur = conn.cursor()
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
    columns = [row[0] for row in cur.fetchall()]
    cur.close()
    return columns

def get_table_row_count(conn, table_name):
    """Get row count for a table without loading all data"""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    cur.close()
    return count

def load_and_merge_table_chunked(conn, base_table_name):
    """Load and merge base table with its _removed counterpart using chunked processing"""
    removed_table_name = f"{base_table_name}_removed"
    
    # Check if both tables exist
    all_tables = get_all_tables(conn)
    base_exists = base_table_name in all_tables
    removed_exists = removed_table_name in all_tables
    
    if not base_exists and not removed_exists:
        print(f"Neither {base_table_name} nor {removed_table_name} exist. Skipping.")
        return None
    
    # Get available columns for both tables
    base_columns = get_available_columns(conn, base_table_name) if base_exists else []
    removed_columns = get_available_columns(conn, removed_table_name) if removed_exists else []
    
    # Find columns that exist in both tables and are in our selected columns
    available_columns = [col for col in SELECTED_COLUMNS if col in base_columns or col in removed_columns]
    
    if not available_columns:
        print(f"No selected columns found in {base_table_name} or {removed_table_name}. Skipping.")
        return None
    
    print(f"Loading {base_table_name} with columns: {available_columns}")
    
    # Create SQLAlchemy engine for better pandas integration
    engine = create_engine(f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}')
    
    # Get row counts first
    base_count = get_table_row_count(conn, base_table_name) if base_exists else 0
    removed_count = get_table_row_count(conn, removed_table_name) if removed_exists else 0
    
    print(f"Base table {base_table_name}: {base_count:,} rows")
    print(f"Removed table {removed_table_name}: {removed_count:,} rows")
    
    # Process in chunks to avoid memory issues
    all_chunks = []
    
    # Process base table
    if base_exists and base_count > 0:
        base_columns_filtered = [col for col in available_columns if col in base_columns]
        if base_columns_filtered:
            print(f"Processing base table in chunks of {CHUNK_SIZE}...")
            offset = 0
            while offset < base_count:
                query = f'SELECT {", ".join(base_columns_filtered)} FROM {base_table_name} LIMIT {CHUNK_SIZE} OFFSET {offset}'
                chunk = pd.read_sql(query, engine)
                if chunk.empty:
                    break
                all_chunks.append(chunk)
                offset += CHUNK_SIZE
                print(f"  Processed {min(offset, base_count):,} / {base_count:,} rows from base table")
                gc.collect()  # Force garbage collection
    
    # Process removed table
    if removed_exists and removed_count > 0:
        removed_columns_filtered = [col for col in available_columns if col in removed_columns]
        if removed_columns_filtered:
            print(f"Processing removed table in chunks of {CHUNK_SIZE}...")
            offset = 0
            while offset < removed_count:
                query = f'SELECT {", ".join(removed_columns_filtered)} FROM {removed_table_name} LIMIT {CHUNK_SIZE} OFFSET {offset}'
                chunk = pd.read_sql(query, engine)
                if chunk.empty:
                    break
                all_chunks.append(chunk)
                offset += CHUNK_SIZE
                print(f"  Processed {min(offset, removed_count):,} / {removed_count:,} rows from removed table")
                gc.collect()  # Force garbage collection
    
    # Combine all chunks
    if all_chunks:
        print("Combining chunks...")
        merged_df = pd.concat(all_chunks, ignore_index=True)
        print(f"Final merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
        
        # Clear chunks from memory
        all_chunks.clear()
        gc.collect()
        
        return merged_df
    else:
        print(f"No data found for {base_table_name}")
        return None

def describe_table(conn, table_name):
    print(f"\n--- Descriptive Analysis for table: {table_name} ---\n")
    
    # Load and merge the table data using chunked processing
    df = load_and_merge_table_chunked(conn, table_name)
    
    if df is None or df.empty:
        print(f"Table {table_name} is empty or not found. Skipping analysis.\n")
        return None, None, None
    
    print(f"Final dataset size: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Get row count
    row_count = len(df)
    print(f"Row count: {row_count:,}\n")
    
    # ===== MESSAGES PER CLASS/CHANNEL =====
    if 'chat_name' in df.columns:
        print("=== MESSAGES PER CLASS/CHANNEL ===")
        messages_per_channel = df['chat_name'].value_counts()
        print(f"Total unique channels: {len(messages_per_channel)}")
        print(f"Average messages per channel: {messages_per_channel.mean():.2f}")
        print(f"Median messages per channel: {messages_per_channel.median():.2f}")
        print(f"Min messages per channel: {messages_per_channel.min()}")
        print(f"Max messages per channel: {messages_per_channel.max()}")
        print("\nTop 10 channels by message count:")
        print(messages_per_channel.head(10))
        print("\nBottom 10 channels by message count:")
        print(messages_per_channel.tail(10))
        print()
    
    # ===== MESSAGE LENGTH ANALYSIS =====
    if 'messagetext' in df.columns:
        print("=== MESSAGE LENGTH ANALYSIS ===")
        # Calculate message lengths (excluding None/NaN) - process in chunks to save memory
        print("Calculating message lengths...")
        df['message_length'] = df['messagetext'].astype(str).str.len()
        message_lengths = df['message_length'].dropna()
        
        if len(message_lengths) > 0:
            print(f"Mean message length: {message_lengths.mean():.2f} characters")
            print(f"Median message length: {message_lengths.median():.2f} characters")
            print(f"Min message length: {message_lengths.min()} characters")
            print(f"Max message length: {message_lengths.max()} characters")
            print(f"Standard deviation: {message_lengths.std():.2f} characters")
            
            # Length distribution
            print(f"\nMessage length distribution:")
            print(f"  Short messages (≤50 chars): {len(message_lengths[message_lengths <= 50])} ({len(message_lengths[message_lengths <= 50])/len(message_lengths)*100:.1f}%)")
            print(f"  Medium messages (51-200 chars): {len(message_lengths[(message_lengths > 50) & (message_lengths <= 200)])} ({len(message_lengths[(message_lengths > 50) & (message_lengths <= 200)])/len(message_lengths)*100:.1f}%)")
            print(f"  Long messages (>200 chars): {len(message_lengths[message_lengths > 200])} ({len(message_lengths[message_lengths > 200])/len(message_lengths)*100:.1f}%)")
            
            # Empty messages
            empty_messages = len(df[df['messagetext'].isna() | (df['messagetext'] == '') | (df['messagetext'] == 'nan')])
            print(f"  Empty messages: {empty_messages} ({empty_messages/len(df)*100:.1f}%)")
        else:
            print("No message text data available")
        print()
    
    # ===== ENGAGEMENT METRICS =====
    print("=== ENGAGEMENT METRICS ===")
    engagement_stats = {}
    
    if 'views' in df.columns:
        views_data = df['views'].dropna()
        if len(views_data) > 0:
            engagement_stats['views'] = {
                'mean': views_data.mean(),
                'median': views_data.median(),
                'max': views_data.max(),
                'total': views_data.sum()
            }
            print(f"Views per message - Mean: {views_data.mean():.2f}, Median: {views_data.median():.2f}, Max: {views_data.max()}")
    
    if 'forwards' in df.columns:
        forwards_data = df['forwards'].dropna()
        if len(forwards_data) > 0:
            engagement_stats['forwards'] = {
                'mean': forwards_data.mean(),
                'median': forwards_data.median(),
                'max': forwards_data.max(),
                'total': forwards_data.sum()
            }
            print(f"Forwards per message - Mean: {forwards_data.mean():.2f}, Median: {forwards_data.median():.2f}, Max: {forwards_data.max()}")
    
    if 'reactions' in df.columns:
        # Enhanced reaction analysis
        def reaction_count(val):
            if pd.isnull(val):
                return 0
            try:
                import json
                d = json.loads(val) if isinstance(val, str) else val
                return sum(d.values()) if isinstance(d, dict) else 0
            except Exception:
                return 0
        
        def extract_reaction_types(val):
            """Extract individual reaction types and their counts"""
            if pd.isnull(val):
                return {}
            try:
                import json
                d = json.loads(val) if isinstance(val, str) else val
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        
        print("Processing reactions...")
        df['reaction_count'] = df['reactions'].apply(reaction_count)
        reaction_data = df['reaction_count'].dropna()
        
        if len(reaction_data) > 0:
            engagement_stats['reactions'] = {
                'mean': reaction_data.mean(),
                'median': reaction_data.median(),
                'max': reaction_data.max(),
                'total': reaction_data.sum()
            }
            print(f"Reactions per message - Mean: {reaction_data.mean():.2f}, Median: {reaction_data.median():.2f}, Max: {reaction_data.max()}")
            
            # Extract all reaction types (process in smaller batches to save memory)
            print("Extracting reaction types...")
            all_reactions = {}
            batch_size = 5000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                reaction_types = batch['reactions'].apply(extract_reaction_types)
                
                for reactions_dict in reaction_types:
                    if isinstance(reactions_dict, dict):
                        for emoji, count in reactions_dict.items():
                            if emoji in all_reactions:
                                all_reactions[emoji] += count
                            else:
                                all_reactions[emoji] = count
                
                if i % 50000 == 0:
                    print(f"  Processed {i:,} / {len(df):,} rows for reactions")
            
            if all_reactions:
                print(f"\n=== REACTION EMOJI ANALYSIS ===")
                print(f"Total unique reaction types: {len(all_reactions)}")
                print(f"Total reactions across all messages: {sum(all_reactions.values()):,}")
                
                # Sort by frequency
                sorted_reactions = sorted(all_reactions.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\nTop 10 most used reactions:")
                for i, (emoji, count) in enumerate(sorted_reactions[:10], 1):
                    percentage = (count / sum(all_reactions.values())) * 100
                    print(f"  {i:2d}. {emoji}: {count:,} ({percentage:.1f}%)")
                
                print(f"\nReaction distribution:")
                print(f"  Messages with reactions: {len(reaction_data[reaction_data > 0])} ({len(reaction_data[reaction_data > 0])/len(df)*100:.1f}%)")
                print(f"  Messages without reactions: {len(reaction_data[reaction_data == 0])} ({len(reaction_data[reaction_data == 0])/len(df)*100:.1f}%)")
                
                # Reaction intensity analysis
                reaction_intensity = reaction_data[reaction_data > 0]
                if len(reaction_intensity) > 0:
                    print(f"\nReaction intensity (for messages with reactions):")
                    print(f"  Average reactions per reactive message: {reaction_intensity.mean():.2f}")
                    print(f"  Median reactions per reactive message: {reaction_intensity.median():.2f}")
                    print(f"  Max reactions on a single message: {reaction_intensity.max()}")
                    
                    # High engagement messages
                    high_engagement = reaction_data[reaction_data >= reaction_intensity.quantile(0.95)]
                    print(f"  High engagement messages (top 5%): {len(high_engagement)} messages with ≥{high_engagement.min()} reactions")
            else:
                print("No reaction data found in the dataset")
    
    if not engagement_stats:
        print("No engagement data available")
    print()
    
    # ===== TEMPORAL PATTERNS =====
    if 'messagedatetime' in df.columns:
        print("=== TEMPORAL PATTERNS ===")
        df['messagedatetime'] = pd.to_datetime(df['messagedatetime'], errors='coerce')
        datetime_data = df['messagedatetime'].dropna()
        
        if len(datetime_data) > 0:
            # Hour of day
            df['hour'] = datetime_data.dt.hour
            hour_counts = df['hour'].value_counts().sort_index()
            peak_hour = hour_counts.idxmax()
            print(f"Peak activity hour: {peak_hour}:00 ({hour_counts.max()} messages)")
            
            # Day of week
            df['day_of_week'] = datetime_data.dt.day_name()
            day_counts = df['day_of_week'].value_counts()
            peak_day = day_counts.idxmax()
            print(f"Peak activity day: {peak_day} ({day_counts.max()} messages)")
            
            # Time span
            time_span = datetime_data.max() - datetime_data.min()
            print(f"Data time span: {time_span.days} days")
            print(f"Date range: {datetime_data.min().date()} to {datetime_data.max().date()}")
        else:
            print("No valid datetime data available")
        print()
    
    # ===== MESSAGE TYPES =====
    print("=== MESSAGE TYPES ===")
    type_stats = {}
    
    if 'post' in df.columns:
        posts = df['post'].sum() if df['post'].dtype in ['bool', 'int64'] else len(df[df['post'] == True])
        type_stats['posts'] = posts
        print(f"Posts: {posts} ({posts/len(df)*100:.1f}%)")
    
    if 'fwd_from' in df.columns:
        forwards = len(df[df['fwd_from'].notna()])
        type_stats['forwards'] = forwards
        print(f"Forwarded messages: {forwards} ({forwards/len(df)*100:.1f}%)")
    
    if 'reply_to' in df.columns:
        replies = len(df[df['reply_to'].notna()])
        type_stats['replies'] = replies
        print(f"Reply messages: {replies} ({replies/len(df)*100:.1f}%)")
    
    if 'media_unread' in df.columns:
        media_messages = len(df[df['media_unread'] == True])
        type_stats['media'] = media_messages
        print(f"Media messages: {media_messages} ({media_messages/len(df)*100:.1f}%)")
    
    if not type_stats:
        print("No message type data available")
    print()
    
    # Basic stats for numeric/date columns
    print("=== BASIC STATISTICS (NUMERIC/DATE COLUMNS) ===")
    print(df.describe(include=["number", "datetime"]))
    print()
    
    # Prepare for combined plots
    messages_per_day_df = None
    messages_per_week_df = None
    channel_counts = None
    
    if 'messagedate' in df.columns:
        df['messagedate'] = pd.to_datetime(df['messagedate'], errors='coerce')
        
        # Daily counts
        messages_per_day = df.groupby(df['messagedate'].dt.date).size()
        messages_per_day_df = messages_per_day.reset_index(name='count')
        messages_per_day_df.rename(columns={'messagedate': 'date'}, inplace=True)
        messages_per_day_df['table'] = table_name
        
        # Weekly counts (non-cumulative)
        df['week_start'] = df['messagedate'].dt.to_period('W').dt.start_time
        messages_per_week = df.groupby('week_start').size()
        messages_per_week_df = messages_per_week.reset_index(name='count')
        messages_per_week_df.rename(columns={'week_start': 'week'}, inplace=True)
        messages_per_week_df['table'] = table_name
        
    if 'chat_name' in df.columns:
        channel_counts = df['chat_name'].value_counts()  # Include all chats
    
    # Clear dataframe from memory
    del df
    gc.collect()
    
    return messages_per_day_df, messages_per_week_df, channel_counts

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
    """Create individual pie charts for each table"""
    for table, counts in channel_counts_dict.items():
        if counts is not None and not counts.empty:
            # Create individual figure for each table
            plt.figure(figsize=(14, 10))
            
            # Show ALL channels individually - no grouping
            plot_counts = counts
            print(f"{table}: Showing all {len(counts)} channels individually")
            
            # Create pie chart with all channels
            wedges, texts, autotexts = plt.pie(plot_counts.values, labels=plot_counts.index, 
                                             autopct='%1.1f%%', startangle=90)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title(f"{table}\n({len(counts)} total channels)", fontsize=14, fontweight='bold')
            
            # Save individual chart
            filename = f"{table}_channels_pie.png"
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved individual pie chart: {filename}")
        else:
            print(f"No data available for {table} pie chart")

def plot_combined_weekly_time_series(messages_per_week_dict):
    """Plot combined weekly message counts for all tables"""
    # Concatenate all non-empty DataFrames
    dfs = [df for df in messages_per_week_dict.values() if df is not None and not df.empty]
    if not dfs:
        print("No data available for weekly time series plot. Skipping plot.")
        return
    all_df = pd.concat(dfs, ignore_index=True)
    
    plt.figure(figsize=(15, 8))
    for table in all_df['table'].unique():
        sub = all_df[all_df['table'] == table]
        plt.plot(sub['week'], sub['count'], marker='o', linewidth=2, markersize=4, label=table)
    
    plt.xlabel('Week Starting Date')
    plt.ylabel('Number of Messages (Weekly Count)')
    plt.title('Weekly Message Counts Over Time (All Tables)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_tables_weekly_messages_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined weekly time series plot: all_tables_weekly_messages_over_time.png")

def plot_weekly_subplots(messages_per_week_dict):
    """Plot weekly message counts in separate subplots for each table"""
    # Filter out empty dataframes
    valid_data = {table: df for table, df in messages_per_week_dict.items() 
                  if df is not None and not df.empty}
    
    if not valid_data:
        print("No data available for weekly subplots. Skipping plot.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (table, df) in enumerate(valid_data.items()):
        if idx < 4:  # Only plot first 4 tables
            ax = axes[idx]
            ax.plot(df['week'], df['count'], marker='o', linewidth=2, 
                   markersize=4, color=colors[idx], label=table)
            
            ax.set_xlabel('Week Starting Date')
            ax.set_ylabel('Number of Messages (Weekly Count)')
            ax.set_title(f'Weekly Message Counts - {table}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add total count annotation
            total_messages = df['count'].sum()
            ax.text(0.02, 0.98, f'Total: {total_messages:,}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(valid_data), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'weekly_messages_subplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved weekly subplots: weekly_messages_subplots.png")

if __name__ == "__main__":
    # Capture all output to save to file
    original_stdout = sys.stdout
    output_buffer = StringIO()
    sys.stdout = output_buffer
    
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        
        print(f"Analyzing target tables: {TARGET_TABLES}")
        print(f"Selected columns: {SELECTED_COLUMNS}")
        
        # Check which tables actually exist
        all_tables = get_all_tables(conn)
        print(f"\nAvailable tables in database: {all_tables}")
        print(f"Looking for tables: {TARGET_TABLES}")
        
        messages_per_day_dict = {}
        messages_per_week_dict = {}
        channel_counts_dict = {}
        table_summaries = {}
        
        for table in TARGET_TABLES:
            print(f"\n{'='*60}")
            print(f"PROCESSING TABLE: {table}")
            print(f"{'='*60}")
            
            messages_per_day_df, messages_per_week_df, channel_counts = describe_table(conn, table)
            messages_per_day_dict[table] = messages_per_day_df
            messages_per_week_dict[table] = messages_per_week_df
            channel_counts_dict[table] = channel_counts
            
            # Store summary for comparison
            if channel_counts is not None:
                # Get row count from the first dataframe that's not None
                row_count = 0
                if messages_per_day_df is not None:
                    row_count = messages_per_day_df['count'].sum()
                elif messages_per_week_df is not None:
                    row_count = messages_per_week_df['count'].sum()
                
                table_summaries[table] = {
                    'total_messages': row_count,
                    'unique_channels': len(channel_counts),
                    'avg_messages_per_channel': channel_counts.mean() if len(channel_counts) > 0 else 0
                }
        
        # ===== CROSS-TABLE COMPARISON SUMMARY =====
        print("\n" + "="*80)
        print("CROSS-TABLE COMPARISON SUMMARY")
        print("="*80)
        
        if table_summaries:
            print(f"{'Table':<30} {'Total Messages':<15} {'Unique Channels':<15} {'Avg Msgs/Channel':<15}")
            print("-" * 80)
            for table, summary in table_summaries.items():
                print(f"{table:<30} {summary['total_messages']:<15,} {summary['unique_channels']:<15} {summary['avg_messages_per_channel']:<15.1f}")
        
        print("\n" + "="*80)
        print("PLOTTING SECTION")
        print("="*80)
        
        # Plot combined charts
        print("Daily messages data:", messages_per_day_dict)
        print("Weekly messages data:", messages_per_week_dict)
        
        # Original daily plots
        plot_combined_time_series(messages_per_day_dict)
        plot_combined_pies(channel_counts_dict)
        
        # New weekly plots
        plot_combined_weekly_time_series(messages_per_week_dict)
        plot_weekly_subplots(messages_per_week_dict)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
    
    # Get the captured output
    output_text = output_buffer.getvalue()
    sys.stdout = original_stdout
    
    # Print to console
    print(output_text)
    
    # Save to text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(OUTPUT_DIR, f"descriptive_analysis_{timestamp}.txt")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\nAnalysis saved to: {output_filename}") 