#!/usr/bin/env python3
"""
Narrative Analysis: Compares narratives between news and community Telegram channels

This script analyzes if propaganda narratives posted in news channels are shared/forwarded 
in community channels, using message similarity and temporal patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime
import json
import math
from tqdm import tqdm
from sqlalchemy import create_engine

# Configure database connection
CONNECTION_STRING = 'postgresql+psycopg2://kiliansprenkamp@localhost:5432/telegram_scraper'
ENGINE = create_engine(CONNECTION_STRING)

def load_data():
    """Load messages from news and community channels in chunks with progress bars"""
    CHUNK_SIZE = 250000
    print("Loading data from PostgreSQL database...")
    
    # First, check all available tables
    with ENGINE.connect() as conn:
        tables_query = """
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        """
        tables = pd.read_sql_query(tables_query, conn)
        print(f"Available tables: {tables['table_name'].tolist()}")
        
        # Use only classified_messages tables
        if 'classified_messages_community' in tables['table_name'].tolist():
            community_table = 'classified_messages_community'
            print("Using classified_messages_community table")
        else:
            print("WARNING: classified_messages_community table not found!")
            community_table = None
            
        if 'classified_messages_news' in tables['table_name'].tolist():
            news_table = 'classified_messages_news'
            print("Using classified_messages_news table")
        else:
            print("WARNING: classified_messages_news table not found!")
            news_table = None
    
    # Function to load data in chunks with progress bar
    def load_in_chunks(query, description):
        # First get the count of rows
        with ENGINE.connect() as conn:
            count_query = f"SELECT COUNT(*) as count FROM ({query}) as subquery"
            total_rows = pd.read_sql_query(count_query, conn).iloc[0]['count']
            
        chunks = []
        num_chunks = math.ceil(total_rows / CHUNK_SIZE)
        
        # Create progress bar
        with tqdm(total=total_rows, desc=description) as pbar:
            for i in range(num_chunks):
                offset = i * CHUNK_SIZE
                chunk_query = f"{query} LIMIT {CHUNK_SIZE} OFFSET {offset}"
                
                with ENGINE.connect() as conn:
                    chunk = pd.read_sql_query(chunk_query, conn)
                    
                chunks.append(chunk)
                pbar.update(len(chunk))
                
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    
    # Load community messages
    if community_table:
        community_query = f"""
            SELECT *
            FROM {community_table}
            WHERE chat_name IN ('https://t.me/readovchat', 'https://t.me/specchatZ')
        """
        df_community = load_in_chunks(community_query, "Loading community messages")
    else:
        df_community = pd.DataFrame()
    
    # Load news messages
    if news_table:
        news_query = f"""
            SELECT *
            FROM {news_table}
        """
        df_news = load_in_chunks(news_query, "Loading news messages")
    else:
        df_news = pd.DataFrame()
    
    # Print column information for debugging
    print("\nCommunity data columns:", df_community.columns.tolist())
    print("\nNews data columns:", df_news.columns.tolist())
    
    # Find similarity columns
    community_similarity_cols = [col for col in df_community.columns if 'similarity' in col.lower()]
    news_similarity_cols = [col for col in df_news.columns if 'similarity' in col.lower()]
    
    print(f"\nFound {len(community_similarity_cols)} similarity columns in community data:")
    for col in community_similarity_cols:
        print(f"  - {col}")
    
    print(f"\nFound {len(news_similarity_cols)} similarity columns in news data:")
    for col in news_similarity_cols:
        print(f"  - {col}")
    
    # Convert datetime columns with progress bar
    print("\nProcessing datetime columns...")
    with tqdm(total=2, desc="Converting datetime columns") as pbar:
        df_community['messagedatetime'] = pd.to_datetime(df_community['messagedatetime'])
        pbar.update(1)
        df_news['messagedatetime'] = pd.to_datetime(df_news['messagedatetime'])
        pbar.update(1)
    
    # Show data ranges
    print(f"Community data: {df_community.shape[0]} messages from {df_community.messagedatetime.min()} to {df_community.messagedatetime.max()}")
    print(f"News data: {df_news.shape[0]} messages from {df_news.messagedatetime.min()} to {df_news.messagedatetime.max()}")
    
    return df_community, df_news, community_similarity_cols, news_similarity_cols

def get_similarity_columns(df):
    """Get actual similarity column names from dataframe"""
    return [col for col in df.columns if 'similarity' in col.lower()]

def classify_narratives(df, similarity_columns, threshold=0.6):
    """Classify messages according to propaganda narratives using actual column names"""
    print(f"\nClassifying narratives for {len(df)} messages...")
    print(f"Using these similarity columns: {similarity_columns}")
    
    # Function to get narratives for a row
    def get_narratives(row):
        return [col.replace("_similarity", "") for col in similarity_columns if col in row and row[col] > threshold]
    
    # Process in chunks with progress bar
    chunk_size = 50000
    total_rows = len(df)
    num_chunks = math.ceil(total_rows / chunk_size)
    
    narratives_list = []
    with tqdm(total=total_rows, desc="Classifying narratives") as pbar:
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk = df.iloc[i:end_idx]
            
            # Apply function to chunk
            chunk_narratives = [get_narratives(row) for _, row in chunk.iterrows()]
            narratives_list.extend(chunk_narratives)
            
            pbar.update(len(chunk))
    
    # Assign calculated narratives to DataFrame
    df['Narratives'] = narratives_list
    
    # Count narratives per message
    with tqdm(desc="Counting narratives", total=1) as pbar:
        df['NarrativeCount'] = df['Narratives'].apply(len)
        pbar.update(1)
    
    # Filter out messages with no narratives
    with tqdm(desc="Filtering messages with narratives", total=1) as pbar:
        df_with_narratives = df[df['NarrativeCount'] > 0].copy()
        pbar.update(1)
    
    print(f"Found {len(df_with_narratives)} messages with at least one narrative")
    
    return df_with_narratives

def analyze_narrative_trends(df_news, df_community, narrative_columns):
    """Plot narrative trends over time for news vs community channels"""
    print("\nAnalyzing narrative trends...")
    
    # Add week column
    df_news['week'] = df_news['messagedatetime'].dt.to_period('W')
    df_community['week'] = df_community['messagedatetime'].dt.to_period('W')
    
    # Plot overall message volume
    plt.figure(figsize=(12, 6))
    df_community.groupby(pd.Grouper(key='messagedatetime', freq='W')).size().plot(label="Community")
    df_news.groupby(pd.Grouper(key='messagedatetime', freq='W')).size().plot(label="News")
    plt.title("Message Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Messages")
    plt.legend()
    plt.tight_layout()
    plt.savefig("narrative_volume_overall.png")
    plt.close()
    
    # Plot narratives
    for narrative in narrative_columns:
        if narrative not in df_news.columns or narrative not in df_community.columns:
            print(f"Skipping {narrative} - not present in both datasets")
            continue
            
        plt.figure(figsize=(12, 6))
        
        narrative_name = narrative.replace("_similarity", "")
        
        # Create news plot
        news_data = df_news[df_news[narrative] > 0.6]
        news_volume = news_data.groupby(pd.Grouper(key='messagedatetime', freq='W')).size()
        news_volume.plot(label="News", color='blue')
        
        # Create community plot
        community_data = df_community[df_community[narrative] > 0.6]
        community_volume = community_data.groupby(pd.Grouper(key='messagedatetime', freq='W')).size()
        community_volume.plot(label="Community", color='red')
        
        plt.title(f"Narrative: {narrative_name}")
        plt.xlabel("Date")
        plt.ylabel("Number of Messages")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"narrative_{narrative_name.replace(' ', '_').replace('_similarity','')}.png")
        plt.close()

def analyze_forwarded_content(df_news, df_community, narrative_columns):
    """Analyze how content is forwarded from news to community channels with progress bars"""
    print("\nAnalyzing forwarded content...")
    
    # Create dictionaries for lookups with progress
    with tqdm(total=1, desc="Creating channel mapping") as pbar:
        peer_id_to_chat = pd.concat(
            [df_news[['peer_id', 'chat_name']], df_community[['peer_id', 'chat_name']]]
        ).drop_duplicates().set_index('peer_id')['chat_name'].to_dict()
        pbar.update(1)
    
    # Add channel names for forwarded messages in chunks
    print("Mapping source channels...")
    df_community['source_channel'] = 'unknown'  # Initialize
    
    # Process in chunks with progress bar
    chunk_size = 50000
    with tqdm(total=len(df_community), desc="Mapping forwarded sources") as pbar:
        for i in range(0, len(df_community), chunk_size):
            end_idx = min(i + chunk_size, len(df_community))
            chunk_indices = df_community.index[i:end_idx]
            
            # Map source channels for this chunk
            df_community.loc[chunk_indices, 'source_channel'] = df_community.loc[chunk_indices, 'fwd_from'].map(peer_id_to_chat).fillna('unknown')
            pbar.update(len(chunk_indices))
    
    # Get news channels
    news_channels = set(df_news['chat_name'].unique())
    
    # Filter to only news-forwarded messages
    with tqdm(total=1, desc="Filtering forwarded messages") as pbar:
        news_forwarded = df_community[df_community['source_channel'].isin(news_channels)]
        pbar.update(1)
    
    print(f"Found {len(news_forwarded)} messages in community channels forwarded from news channels")
    
    # Show top news sources
    with tqdm(total=1, desc="Analyzing top sources") as pbar:
        source_counts = news_forwarded['source_channel'].value_counts()
        pbar.update(1)
        
    print("\nTop news sources forwarded to community channels:")
    print(source_counts.head(10))
    
    # Output examples with high narrative scores
    print(f"\nAnalyzing top examples for {len(narrative_columns)} narratives...")
    
    with tqdm(total=len(narrative_columns), desc="Finding top examples") as pbar:
        for narrative in narrative_columns:
            if narrative not in df_community.columns:
                print(f"Skipping {narrative} - not in community data")
                pbar.update(1)
                continue
                
            narrative_name = narrative.replace("_similarity", "")
            print(f"\nTop examples of '{narrative_name}' forwarded from news channels:")
            
            try:
                top_examples = news_forwarded.sort_values(narrative, ascending=False).head(3)
                for _, row in top_examples.iterrows():
                    print(f"- Source: {row['source_channel']}")
                    print(f"  Score: {row[narrative]:.4f}")
                    print(f"  Text: {row['messagetext'][:100]}...")
                    print()
            except Exception as e:
                print(f"Error analyzing {narrative}: {e}")
            
            pbar.update(1)
    
    return news_forwarded

def calculate_cross_correlation(df_news, df_community):
    """Calculate cross-correlation between news and community for each narrative"""
    print("\nCalculating cross-correlation between news and community narratives...")
    
    # Prepare data
    df_news_exploded = df_news.explode('Narratives')
    df_community_exploded = df_community.explode('Narratives')
    
    # Aggregate by week
    news_counts = df_news_exploded.groupby(['week', 'Narratives']).size().reset_index(name='Count')
    community_counts = df_community_exploded.groupby(['week', 'Narratives']).size().reset_index(name='Count')
    
    # Pivot to get counts per narrative per week
    news_pivot = news_counts.pivot(index='week', columns='Narratives', values='Count').fillna(0)
    community_pivot = community_counts.pivot(index='week', columns='Narratives', values='Count').fillna(0)
    
    # Get common narratives
    common_narratives = sorted(set(news_pivot.columns) & set(community_pivot.columns))
    print(f"Found {len(common_narratives)} common narratives for correlation analysis")
    
    # Normalize to get proportions
    news_normalized = news_pivot.div(news_pivot.sum(axis=1), axis=0).fillna(0)
    community_normalized = community_pivot.div(community_pivot.sum(axis=1), axis=0).fillna(0)
    
    # Calculate correlation
    correlation_results = pd.Series(index=common_narratives)
    lag_results = {}
    
    for narrative in common_narratives:
        if narrative in news_normalized.columns and narrative in community_normalized.columns:
            # Calculate correlation at different lags (0-3 weeks)
            correlations = []
            for lag in range(4):
                shifted_news = news_normalized[narrative].shift(lag)
                corr = shifted_news.corr(community_normalized[narrative])
                correlations.append((lag, corr))
            
            # Find max correlation
            max_corr = max(correlations, key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0)
            correlation_results[narrative] = max_corr[1]
            lag_results[narrative] = max_corr[0]
    
    # Plot correlation results
    plt.figure(figsize=(14, 7))
    correlation_results.sort_values().plot(kind='barh')
    plt.title('Maximum Correlation Between News and Community Narratives')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig("narrative_correlation.png")
    plt.close()
    
    print("\nCorrelation Results (with optimal lag):")
    for narrative in correlation_results.index:
        print(f"{narrative}: {correlation_results[narrative]:.4f} (lag: {lag_results[narrative]} weeks)")
    
    return correlation_results, lag_results

def perform_granger_causality(df_news, df_community):
    """Perform Granger causality test to see if news channels cause community narratives"""
    print("\nPerforming Granger causality tests...")
    
    # Prepare data - same as in cross_correlation
    df_news_exploded = df_news.explode('Narratives')
    df_community_exploded = df_community.explode('Narratives')
    
    news_counts = df_news_exploded.groupby(['week', 'Narratives']).size().reset_index(name='Count')
    community_counts = df_community_exploded.groupby(['week', 'Narratives']).size().reset_index(name='Count')
    
    news_pivot = news_counts.pivot(index='week', columns='Narratives', values='Count').fillna(0)
    community_pivot = community_counts.pivot(index='week', columns='Narratives', values='Count').fillna(0)
    
    common_narratives = sorted(set(news_pivot.columns) & set(community_pivot.columns))
    
    # For each narrative, test if news Granger-causes community
    results = {}
    for narrative in common_narratives:
        if narrative in news_pivot.columns and narrative in community_pivot.columns:
            news_series = news_pivot[narrative]
            community_series = community_pivot[narrative]
            
            # Align series and drop NaNs
            combined = pd.concat([news_series, community_series], axis=1).dropna()
            combined.columns = ['News', 'Community']
            
            if len(combined) > 8:  # Need at least 8 observations for lag of 4
                try:
                    # Perform Granger test with max lag of 4 weeks
                    granger_result = grangercausalitytests(combined, maxlag=4, verbose=False)
                    
                    # Extract p-values for each lag
                    p_values = [granger_result[i+1][0]['ssr_ftest'][1] for i in range(4)]
                    
                    # Find minimum p-value and corresponding lag
                    min_p = min(p_values)
                    min_lag = p_values.index(min_p) + 1
                    
                    results[narrative] = {
                        'min_p_value': min_p,
                        'lag': min_lag,
                        'significant': min_p < 0.05
                    }
                except Exception as e:
                    print(f"Error calculating Granger causality for {narrative}: {e}")
                    results[narrative] = {
                        'min_p_value': float('nan'),
                        'lag': float('nan'),
                        'significant': False,
                        'error': True
                    }
            else:
                results[narrative] = {
                    'min_p_value': float('nan'),
                    'lag': float('nan'),
                    'significant': False,
                    'error': 'Not enough data'
                }
    
    # Print results
    print("\nGranger Causality Results (News → Community):")
    for narrative, result in results.items():
        if result.get('error', False) is True:
            print(f"{narrative}: Error in calculation")
        elif isinstance(result.get('error', False), str):
            print(f"{narrative}: {result['error']}")
        else:
            sig_marker = "*" if result['significant'] else ""
            print(f"{narrative}: p={result['min_p_value']:.4f} at lag {result['lag']} weeks {sig_marker}")
    
    # Count significant results
    significant_count = sum(1 for r in results.values() if r.get('significant', False))
    print(f"\n{significant_count} out of {len(results)} narratives show significant Granger causality (p<0.05)")
    
    return results

def analyze_text_similarity(df_news, df_community):
    """Analyze textual similarity between news and community messages using chunked processing"""
    print("\nAnalyzing textual similarity between news and community messages...")
    
    # Ensure we have the necessary columns
    required_cols = ['messagetext', 'peer_id', 'chat_name', 'fwd_from']
    for col in required_cols:
        if col not in df_news.columns:
            print(f"WARNING: Column '{col}' missing from news data. Using limited analysis.")
        if col not in df_community.columns:
            print(f"WARNING: Column '{col}' missing from community data. Using limited analysis.")
    
    # Sample for performance
    sample_size = min(5000, len(df_news))
    df_news_sample = df_news.sample(sample_size) if len(df_news) > sample_size else df_news
    
    # Check for exact matches with progress bar (if messagetext column exists)
    exact_matches = 0
    if 'messagetext' in df_news.columns and 'messagetext' in df_community.columns:
        print("Checking for exact matches...")
        news_texts = set()
        community_texts = set()
        
        # Process news texts in chunks
        with tqdm(total=len(df_news_sample), desc="Processing news texts") as pbar:
            for i in range(0, len(df_news_sample), 10000):
                chunk = df_news_sample.iloc[i:i+10000]
                news_texts.update(chunk['messagetext'])
                pbar.update(len(chunk))
        
        # Process community texts in chunks
        with tqdm(total=len(df_community), desc="Processing community texts") as pbar:
            for i in range(0, len(df_community), 10000):
                chunk = df_community.iloc[i:i+10000]
                community_texts.update(chunk['messagetext'])
                pbar.update(len(chunk))
        
        # Find intersection
        with tqdm(total=1, desc="Finding exact matches") as pbar:
            exact_matches = len(news_texts.intersection(community_texts))
            pbar.update(1)
        
        print(f"Found {exact_matches} exact text matches between news and community channels")
    else:
        print("Skipping exact match analysis - messagetext column not available in both datasets")
    
    # Check for forwarded messages from news channels
    forwarded_count = 0
    if all(col in df_news.columns and col in df_community.columns for col in ['peer_id', 'chat_name', 'fwd_from']):
        print("Analyzing forwarded content...")
        
        # Create peer_id to chat_name dictionary
        with tqdm(total=1, desc="Building peer_id mapping") as pbar:
            peer_id_to_chat = df_news[['peer_id', 'chat_name']].drop_duplicates().set_index('peer_id')['chat_name'].to_dict()
            pbar.update(1)
        
        # Map source channels in chunks
        with tqdm(total=len(df_community), desc="Mapping source channels") as pbar:
            df_community['source_channel'] = 'unknown'  # Initialize with default
            for i in range(0, len(df_community), 50000):
                chunk = df_community.iloc[i:i+50000]
                df_community.loc[chunk.index, 'source_channel'] = chunk['fwd_from'].map(peer_id_to_chat).fillna('unknown')
                pbar.update(len(chunk))
        
        # Get news channels
        news_channels = set(df_news['chat_name'].unique())
        
        # Find forwarded messages
        with tqdm(total=1, desc="Finding forwarded messages") as pbar:
            forwarded_from_news = df_community[df_community['source_channel'].isin(news_channels)]
            forwarded_count = len(forwarded_from_news)
            pbar.update(1)
        
        print(f"Found {forwarded_count} messages in community channels forwarded from news channels")
        
        # Sample of forwarded messages for inspection
        if forwarded_count > 0:
            print("\nSample of forwarded messages from news to community channels:")
            sample = forwarded_from_news.sample(min(5, forwarded_count))
            for _, row in sample.iterrows():
                print(f"- Source: {row['source_channel']}")
                if 'messagetext' in row:
                    print(f"  Text: {row['messagetext'][:100]}...")
                print()
    else:
        print("Skipping forwarded content analysis - required columns not available in both datasets")
    
    return {
        'exact_matches': exact_matches,
        'forwarded_count': forwarded_count
    }

def main():
    """Main function to analyze narratives between news and community channels"""
    print("Starting narrative analysis between news and community channels...")
    
    # Load data
    df_community, df_news, community_similarity_cols, news_similarity_cols = load_data()
    
    # Use common similarity columns  
    common_similarity_cols = list(set(community_similarity_cols) & set(news_similarity_cols))
    print(f"\nCommon similarity columns between news and community: {common_similarity_cols}")
    
    # If no common columns, use all available
    if not common_similarity_cols:
        print("No common similarity columns found. Using all available columns.")
        all_similarity_cols = list(set(community_similarity_cols) | set(news_similarity_cols))
        common_similarity_cols = all_similarity_cols
        
    # Sort columns for consistent output
    common_similarity_cols.sort()
    
    # Classify narratives
    df_community_with_narratives = classify_narratives(df_community, community_similarity_cols)
    df_news_with_narratives = classify_narratives(df_news, news_similarity_cols)
    
    # Analyze trends
    analyze_narrative_trends(df_news_with_narratives, df_community_with_narratives, common_similarity_cols)
    
    # Analyze forwarded content
    forwarded_content = analyze_forwarded_content(df_news_with_narratives, df_community_with_narratives, common_similarity_cols)
    
    # Calculate cross-correlation
    correlation_results, lag_results = calculate_cross_correlation(df_news_with_narratives, df_community_with_narratives)
    
    # Perform Granger causality tests
    granger_results = perform_granger_causality(df_news_with_narratives, df_community_with_narratives)
    
    # Analyze text similarity
    similarity_results = analyze_text_similarity(df_news_with_narratives, df_community_with_narratives)
    
    print("\nAnalysis complete!")
    print(f"Generated visualizations: narrative_volume_overall.png and individual narrative plots")
    
    return {
        'correlation': correlation_results,
        'granger': granger_results,
        'similarity': similarity_results
    }

if __name__ == "__main__":
    main()