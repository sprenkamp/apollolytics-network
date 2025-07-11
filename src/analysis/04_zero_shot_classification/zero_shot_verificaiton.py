import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

# DRY RUN MODE: Only process first 10 rows per narrative above 0.85
DRY_RUN = False

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Load messages
df = pd.read_sql("SELECT * FROM relevant_classified_narrative_results", engine)

# Helper to sanitize narrative keys for column names
def sanitize_key(key):
    sanitized = re.sub(r'[ \-/]', '_', key)
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)
    return sanitized.lower()

# Load narratives from Excel 
def load_narratives_from_excel(file_path):
    xls = pd.ExcelFile(file_path)
    narratives = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        narratives[sheet_name] = dict(zip(df['NarrativeName'], df['NarrativeOriginalLanguage']))
    return narratives

NARRATIVES_FILE_PATH = 'data/propaganda_narratives/propagandaNarratives.xlsx'
narratives = load_narratives_from_excel(NARRATIVES_FILE_PATH)

# Zero-shot classifier
def gpt_zero_shot_classify(message, narrative, model="gpt-4.1-mini"):
    prompt = (
        f"Does the following message express or support the narrative below? "
        f"Reply only with 'yes' or 'no'.\n\n"
        f"Message: {message}\n\n"
        f"Narrative: {narrative}"
    )
    # print(f"Prompt: {prompt}")
    try:
        response = client.chat.completions.create(model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0)
        answer = response.choices[0].message.content.strip().lower()
        return 1 if answer.startswith('yes') else 0
    except Exception as e:
        print(f"OpenAI error: {e}. Returning 0.")
        return 0

def classify_messages_parallel(messages, narrative_text, max_workers=10):
    results = [None] * len(messages)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(gpt_zero_shot_classify, msg, narrative_text): i
            for i, msg in enumerate(messages)
        }
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_idx), total=len(messages), desc="Classifying in parallel"):
            i = future_to_idx[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error in classification: {e}")
                results[i] = 0
    return results

# For each message, classify only with relevant narratives
country_to_sheet = {"Russia": "RussianNarratives", "Ukraine": "UkrainianNarratives"}
tqdm.pandas()
for country, sheet in country_to_sheet.items():
    print(f"Processing {country} {sheet}")
    relevant_mask = df['country'] == country
    for narrative_key, narrative_text in narratives.get(sheet, {}).items():
        print(f"Processing {narrative_key} {narrative_text}")
        col_name = f"gpt_match_{sanitize_key(narrative_key)}"
        similarity_col = f"{narrative_key}_similarity".lower()
        if similarity_col not in df.columns:
            continue
        # Only select rows above 0.85 for this narrative
        mask = relevant_mask & (df[similarity_col] > 0.85)
        idx = df[mask].index
        if DRY_RUN:
            idx = idx[:1000]  # Only first 10 rows
        print(f"Classifying for {country} narrative: {narrative_key} ({len(idx)} rows)")
        messages = df.loc[idx, 'messagetext'].tolist()
        results = classify_messages_parallel(messages, narrative_text, max_workers=10)  # Adjust max_workers as needed
        df.loc[idx, col_name] = results

# Fill all NaN values in gpt_match_ columns with 0
gpt_match_cols = [col for col in df.columns if col.startswith("gpt_match_")]
df[gpt_match_cols] = df[gpt_match_cols].fillna(0)

# Write to new Postgres table
df.to_sql("gpt_zero_shot_narrative_results", engine, if_exists='replace', index=False)
print("Saved results to gpt_zero_shot_narrative_results table.")
