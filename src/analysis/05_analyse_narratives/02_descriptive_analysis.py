import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json

"""
Script: 02_descriptive_analysis.py

This script performs descriptive analysis of narrative propagation for messages above 0.875 similarity threshold.
It analyzes narrative statistics per country (Russia/Ukraine) and chat type (Channels/Groups).

- Input: relevant_classified_narrative_results table with narrative similarity scores
- Output: Descriptive statistics table showing narrative patterns

Usage:
    python 02_descriptive_analysis.py
"""

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

SIMILARITY_THRESHOLD = 0.875

engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def get_narrative_columns(df):
    return [col for col in df.columns if col.lower().endswith('_similarity')]

def get_country_narratives():
    russian_narratives = [
        'denazificationofukraine_similarity',
        'protectionofrussianspeakers_similarity', 
        'natoexpansionthreat_similarity',
        'biolabsconspiracy_similarity',
        'historicalunity_similarity',
        'westernrussophobia_similarity',
        'sanctionsaseconomicwarfare_similarity',
        'legitimizingannexedterritories_similarity',
        'discreditingukraineleadership_similarity'
    ]
    ukrainian_narratives = [
        'putinsdeath_similarity',
        'russiascollapse_similarity',
        'nordstreampipelinesabotage_similarity',
        'heroicmyths_similarity',
        'optimismstrategy_similarity',
        'notsidingwithukraine_similarity',
        'ukraineagainstnewfascism_similarity',
        'cannonfodder_similarity',
        'truthvsrussianlies_similarity'
    ]
    return russian_narratives, ukrainian_narratives

def extract_reactions(val):
    if pd.isnull(val):
        return {}
    try:
        d = json.loads(val) if isinstance(val, str) else val
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def analyze_narrative_stats(df, country, chat_type):
    # Filter data
    country_df = df[df['country'] == country].copy()
    if 'type' in country_df.columns:
        if chat_type == 'Channels':
            filtered_df = country_df[country_df['type'].str.lower() == 'channel']
        else:
            filtered_df = country_df[country_df['type'].str.lower() == 'group']
    else:
        if chat_type == 'Channels':
            filtered_df = country_df[country_df['chat_name'].str.contains('channel|канал', case=False, na=False)]
        else:
            filtered_df = country_df[~country_df['chat_name'].str.contains('channel|канал', case=False, na=False)]
    if filtered_df.empty:
        return None
    russian_narratives, ukrainian_narratives = get_country_narratives()
    if country == 'Russia':
        narrative_cols = russian_narratives
    else:
        narrative_cols = ukrainian_narratives
    available_narratives = [col for col in narrative_cols if col in filtered_df.columns]
    narrative_stats = {}
    for narrative_col in available_narratives:
        narrative_name = narrative_col.replace('_similarity', '').replace('_', ' ')
        # Only messages above threshold for this narrative
        mask = filtered_df[narrative_col] > SIMILARITY_THRESHOLD
        narrative_df = filtered_df[mask].copy()
        if narrative_df.empty:
            stats = {
                'total_messages': 0,
                'mean_msg_length': 0.0,
                'mean_views': 0.0,
                'mean_forwards': 0.0,
                'max_forwards': 0.0,
                'mean_reactions': 0.0,
                'max_reactions': 0.0,
                'top10_emojis': ''
            }
        else:
            # Message length
            narrative_df['msg_length'] = narrative_df['messagetext'].astype(str).str.len()
            # Views
            mean_views = narrative_df['views'].mean() if 'views' in narrative_df.columns else 0.0
            # Forwards
            mean_forwards = narrative_df['forwards'].mean() if 'forwards' in narrative_df.columns else 0.0
            max_forwards = narrative_df['forwards'].max() if 'forwards' in narrative_df.columns else 0.0
            # Reactions
            if 'reactions' in narrative_df.columns:
                narrative_df['reaction_count'] = narrative_df['reactions'].apply(lambda x: sum(extract_reactions(x).values()))
                mean_reactions = narrative_df['reaction_count'].mean()
                max_reactions = narrative_df['reaction_count'].max()
                # Top 10 emojis
                all_emojis = {}
                for val in narrative_df['reactions']:
                    for emoji, count in extract_reactions(val).items():
                        all_emojis[emoji] = all_emojis.get(emoji, 0) + count
                top10 = sorted(all_emojis.items(), key=lambda x: x[1], reverse=True)[:10]
                top10_str = ', '.join([f'{emoji}: {count}' for emoji, count in top10])
            else:
                mean_reactions = 0.0
                max_reactions = 0.0
                top10_str = ''
            stats = {
                'total_messages': len(narrative_df),
                'mean_msg_length': narrative_df['msg_length'].mean(),
                'mean_views': mean_views,
                'mean_forwards': mean_forwards,
                'max_forwards': max_forwards,
                'mean_reactions': mean_reactions,
                'max_reactions': max_reactions,
                'top10_emojis': top10_str
            }
        narrative_stats[narrative_name] = stats
    return narrative_stats

def create_descriptive_table(df):
    countries = ['Russia', 'Ukraine']
    chat_types = ['Channels', 'Groups']
    russian_narratives, ukrainian_narratives = get_country_narratives()
    stat_rows = [
        'total_messages',
        'mean_msg_length',
        'mean_views',
        'mean_forwards',
        'max_forwards',
        'mean_reactions',
        'max_reactions',
        'top10_emojis'
    ]
    columns = []
    results = {}
    for country in countries:
        for chat_type in chat_types:
            key = f"{country}_{chat_type}"
            stats = analyze_narrative_stats(df, country, chat_type)
            results[key] = stats
    # Build columns: (Country, ChatType, Narrative)
    all_narratives = []
    for country in countries:
        if country == 'Russia':
            narrative_names = [col.replace('_similarity', '').replace('_', ' ') for col in russian_narratives]
        else:
            narrative_names = [col.replace('_similarity', '').replace('_', ' ') for col in ukrainian_narratives]
        for chat_type in chat_types:
            for narrative in narrative_names:
                columns.append((country, chat_type, narrative))
                all_narratives.append((country, chat_type, narrative))
    # Build data: rows are stats, columns are (country, chat_type, narrative)
    data = {stat: [] for stat in stat_rows}
    for (country, chat_type, narrative) in all_narratives:
        key = f"{country}_{chat_type}"
        if results[key] and narrative in results[key]:
            for stat in stat_rows:
                data[stat].append(results[key][narrative][stat])
        else:
            for stat in stat_rows:
                data[stat].append('')
    df_table = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(columns, names=['Country', 'Chat_Type', 'Narrative'])).T
    return df_table

def main():
    print("Starting narrative descriptive analysis...")
    df = pd.read_sql("SELECT * FROM relevant_classified_narrative_results", engine)
    similarity_cols = get_narrative_columns(df)
    if similarity_cols:
        mask = (df[similarity_cols] > SIMILARITY_THRESHOLD).any(axis=1)
        df = df[mask]
        print(f"Filtered to {len(df)} rows with narrative similarity > {SIMILARITY_THRESHOLD}")
    descriptive_table = create_descriptive_table(df)
    os.makedirs('narrative_analysis_output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Wide format
    csv_filename = f'narrative_analysis_output/narrative_descriptive_table_{timestamp}.csv'
    descriptive_table.to_csv(csv_filename)
    excel_filename = f'narrative_analysis_output/narrative_descriptive_table_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        descriptive_table.to_excel(writer, sheet_name='Narrative_Statistics')
    # Inverted format
    inverted_table = descriptive_table.T
    inverted_csv = f'narrative_analysis_output/narrative_descriptive_table_inverted_{timestamp}.csv'
    inverted_table.to_csv(inverted_csv)
    inverted_excel = f'narrative_analysis_output/narrative_descriptive_table_inverted_{timestamp}.xlsx'
    with pd.ExcelWriter(inverted_excel, engine='openpyxl') as writer:
        inverted_table.to_excel(writer, sheet_name='Narrative_Statistics_Inverted')
    print(f"\nResults saved to:")
    print(f"  Wide CSV: {csv_filename}")
    print(f"  Wide Excel: {excel_filename}")
    print(f"  Inverted CSV: {inverted_csv}")
    print(f"  Inverted Excel: {inverted_excel}")
    print("First few rows (wide):")
    print(descriptive_table.head())
    print("First few rows (inverted):")
    print(inverted_table.head())

if __name__ == "__main__":
    main()
