import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

tqdm.pandas()

# Load environment variables
load_dotenv()
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "telegram_scraper")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

SIMILARITY_THRESHOLD = 0.875

SENTIMENT_CLASSES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

engine = create_engine(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def get_narrative_columns(df):
    return [col for col in df.columns if col.lower().endswith('_similarity')]

def main():
    print("Loading messages with narrative similarity > 0.875...")
    df = pd.read_sql("SELECT * FROM relevant_classified_narrative_results", engine)
    similarity_cols = get_narrative_columns(df)
    mask = (df[similarity_cols] > SIMILARITY_THRESHOLD).any(axis=1)
    df = df[mask].copy()
    print(f"Loaded {len(df)} messages.")

    needs_classification = not ({"sentiment_label", "sentiment_score"} <= set(df.columns))
    if needs_classification:
        print("Sentiment columns not found. Running sentiment classification...")
        from transformers import pipeline
        sentiment_pipe = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

        def classify_sentiment(text):
            if not isinstance(text, str) or not text.strip():
                return {"label": None, "score": None}
            try:
                result = sentiment_pipe(text[:512])[0]
                label = result["label"]
                if label not in SENTIMENT_CLASSES:
                    label = None
                return {"label": label, "score": result["score"]}
            except Exception:
                return {"label": None, "score": None}

        from tqdm import tqdm
        sentiment_labels = []
        sentiment_scores = []
        for text in tqdm(df["messagetext"], desc="Sentiment analysis"):
            result = classify_sentiment(text)
            sentiment_labels.append(result["label"])
            sentiment_scores.append(result["score"])
        df["sentiment_label"] = sentiment_labels
        df["sentiment_score"] = sentiment_scores

        print("Writing results to table: relevant_classified_narrative_results")
        df.to_sql("relevant_classified_narrative_results", engine, if_exists="replace", index=False)
    else:
        print("Sentiment columns found. Skipping classification.")

    
def plot_sentiment_matrix(df, output_dir="sentiment_barplots_matrix"):
    os.makedirs(output_dir, exist_ok=True)
    countries = ['Russia', 'Ukraine']
    chat_types = ['channel', 'group']
    narrative_map = {
        'Russia': [
            'denazificationofukraine_similarity',
            'protectionofrussianspeakers_similarity',
            'natoexpansionthreat_similarity',
            'biolabsconspiracy_similarity',
            'historicalunity_similarity',
            'westernrussophobia_similarity',
            'sanctionsaseconomicwarfare_similarity',
            'legitimizingannexedterritories_similarity',
            'discreditingukraineleadership_similarity'
        ],
        'Ukraine': [
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
    }
    for country in countries:
        narratives = narrative_map[country]
        fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharey=True)
        for i, narrative_col in enumerate(narratives):
            for j, chat_type in enumerate(chat_types):
                row = i // 3
                col = (i % 3) * 2 + j
                ax = axes[row, col]
                mask = (df['country'] == country) & (df['type'].str.lower() == chat_type) & (df[narrative_col] > SIMILARITY_THRESHOLD)
                subdf = df[mask]
                counts = subdf['sentiment_label'].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)
                counts.plot(kind='bar', ax=ax, color=['#d62728', '#ff7f0e', '#cccccc', '#2ca02c', '#1f77b4'])
                ax.set_title(f"{narrative_col.replace('_similarity','').replace('_',' ')}\n{chat_type.title()}", fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels(SENTIMENT_CLASSES, rotation=30)
        plt.tight_layout()
        plt.suptitle(f"Sentiment Distribution per Narrative ({country})", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.92)
        fname = f"{output_dir}/sentiment_matrix_{country.lower()}.png"
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

def plot_sentiment_bars_side_by_side(df, output_dir="sentiment_barplots_sidebyside"):
    os.makedirs(output_dir, exist_ok=True)
    countries = ['Russia', 'Ukraine']
    chat_types = ['channel', 'group']
    narrative_map = {
        'Russia': [
            'denazificationofukraine_similarity',
            'protectionofrussianspeakers_similarity',
            'natoexpansionthreat_similarity',
            'biolabsconspiracy_similarity',
            'historicalunity_similarity',
            'westernrussophobia_similarity',
            'sanctionsaseconomicwarfare_similarity',
            'legitimizingannexedterritories_similarity',
            'discreditingukraineleadership_similarity'
        ],
        'Ukraine': [
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
    }
    for country in countries:
        narratives = narrative_map[country]
        for narrative_col in narratives:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            for idx, chat_type in enumerate(chat_types):
                mask = (
                    (df['country'] == country) &
                    (df['type'].str.lower() == chat_type) &
                    (df[narrative_col] > SIMILARITY_THRESHOLD)
                )
                subdf = df[mask]
                counts = subdf['sentiment_label'].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)
                counts.plot(kind='bar', ax=axes[idx], color=['#d62728', '#ff7f0e', '#cccccc', '#2ca02c', '#1f77b4'])
                axes[idx].set_title(f"{chat_type.title()}s")
                axes[idx].set_xlabel("Sentiment")
                axes[idx].set_ylabel("Message Count" if idx == 0 else "")
                axes[idx].set_xticklabels(SENTIMENT_CLASSES, rotation=30)
            plt.suptitle(f"{country} - {narrative_col.replace('_similarity','').replace('_',' ')}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"{output_dir}/{country}_{narrative_col.replace('_similarity','')}_sentiment.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {fname}")

def plot_stacked_sentiment_bars(df, output_dir="analysis_output/sentiment_stacked_barplots"):
    os.makedirs(output_dir, exist_ok=True)
    countries = ['Russia', 'Ukraine']
    chat_types = ['channel', 'group']
    narrative_map = {
        'Russia': [
            'denazificationofukraine_similarity',
            'protectionofrussianspeakers_similarity',
            'natoexpansionthreat_similarity',
            'biolabsconspiracy_similarity',
            'historicalunity_similarity',
            'westernrussophobia_similarity',
            'sanctionsaseconomicwarfare_similarity',
            'legitimizingannexedterritories_similarity',
            'discreditingukraineleadership_similarity'
        ],
        'Ukraine': [
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
    }
    colors = ['#d62728', '#ff7f0e', '#cccccc', '#2ca02c', '#1f77b4']
    for country in countries:
        narratives = narrative_map[country]
        bar_width = 0.35
        x = np.arange(len(narratives))
        fig, ax = plt.subplots(figsize=(18, 7))
        # Prepare data: for each narrative, for each chat type, get sentiment distribution (normalized)
        for idx, chat_type in enumerate(chat_types):
            all_props = []
            for narrative_col in narratives:
                mask = (
                    (df['country'] == country) &
                    (df['type'].str.lower() == chat_type) &
                    (df[narrative_col] > SIMILARITY_THRESHOLD)
                )
                subdf = df[mask]
                counts = subdf['sentiment_label'].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)
                total = counts.sum()
                if total > 0:
                    props = counts / total
                else:
                    props = counts  # all zeros
                all_props.append(props.values)
            all_props = np.array(all_props).T  # shape: (5, 9)
            # Stacked bar
            bottoms = np.zeros(len(narratives))
            for i, sentiment in enumerate(SENTIMENT_CLASSES):
                ax.bar(
                    x + (idx - 0.5) * bar_width,
                    all_props[i],
                    bar_width,
                    bottom=bottoms,
                    color=colors[i],
                    label=sentiment if (idx == 0) else None
                )
                bottoms += all_props[i]
        # X-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_similarity','').replace('_',' ') for n in narratives], rotation=30, ha='right')
        ax.set_xlabel("Narrative")
        ax.set_ylabel("Proportion of Messages")
        ax.set_title(f"Relative Sentiment Distribution per Narrative ({country})")
        ax.legend(SENTIMENT_CLASSES, title="Sentiment", bbox_to_anchor=(1.01, 1), loc='upper left')
        # Add group labels
        for idx, chat_type in enumerate(chat_types):
            xpos = x + (idx - 0.5) * bar_width
            for xi in xpos:
                ax.text(xi, 1.02, chat_type.title(), ha='center', va='bottom', fontsize=8, rotation=90)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        fname = f"{output_dir}/sentiment_stacked_{country.lower()}.png"
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

if __name__ == "__main__":
    main()
