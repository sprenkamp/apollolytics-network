import pandas as pd
import numpy as np
import json

def total_interactions(x):
    reactions_total = 0
    for _, value in json.loads(x.replace("'", '"')).items():
        reactions_total += value
    return reactions_total

def min_max_normalization(col, df):
    return (df[col] - df[col].min()) / (df[col].max() - df[col].min())

def calculate_absolute_engagement(row):
    # Absolute engagement as the ratio of interactions (reactions + forwards) to views
    if row.views == 0:
        return 0
    return (row.reaction_nb + row.forwards) / row.views

def calculate_normalized_engagement(row):
    # Normalized engagement uses the normalized values
    if row.views_norm == 0:
        return 0
    return (row.reaction_nb_norm + row.forwards_norm) / row.views_norm

def calculate_popularity_score(row):
    return np.mean([row.views, row.forwards, row.reaction_nb])

def calculate_popularity_score_norm(row):
    return np.mean([row.views_norm, row.forwards_norm, row.reaction_nb_norm])


def main():
    # Load data
    df = pd.read_csv('data/telegram/messages_scraped.csv')
    
    # Calculate total interactions
    df['reaction_nb'] = df.reactions.apply(lambda x: total_interactions(x))
    
    # Normalize the columns
    df["views_norm"] = min_max_normalization("views", df)
    df["forwards_norm"] = min_max_normalization("forwards", df)
    df["reaction_nb_norm"] = min_max_normalization("reaction_nb", df)
    
    # Calculate absolute and normalized engagement scores
    df['engagement'] = df.apply(lambda x: calculate_absolute_engagement(x), axis=1)
    df['engagement_norm'] = df.apply(lambda x: calculate_normalized_engagement(x), axis=1)
    df['popularity_score'] = df.apply(lambda x: calculate_popularity_score(x), axis=1)
    df['popularity_score_norm'] = df.apply(lambda x: calculate_popularity_score_norm(x), axis=1)
    
    # Save the result to a new CSV file
    df.to_csv('data/telegram/messages_scraped.csv', index=False)

if __name__ == "__main__":
    main()
