import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))

def get_fwd_name(id, original_df, links_df):
    res = original_df[original_df.peer_id == id].chat
    if len(res) < 1:
        res = links_df[links_df.id == id].chat_name
        if len(res) < 1:
            return None
        else:
            return res.values[0]
    else:
        return res.values[0]
    

def ner_message(x, ner, labels):
    return [[ent.text for ent in sent.ents if ent.label_ in labels] for sent in ner(x).sents]

def total_interactions(x):
    reactions_total = 0
    for _, value in json.loads(x.replace("'", '"')).items():
        reactions_total += value
    return reactions_total

def weighted_popularity_score(row):
    return np.mean([row.views_norm, row.forwards_norm, row.reaction_nb_norm])

def min_max_normalization(col, df):
    return (df[col]-df[col].min())/(df[col].max()-df[col].min())

def remove_stopwords(m, stopwords):
    word_tokens = word_tokenize(m)
    return ' '.join([w for w in word_tokens if not w.lower() in stopwords])

def compute_popularity(df):
    df['reaction_nb'] = df.reactions.apply(lambda x: total_interactions(x))
    df["views_norm"] = min_max_normalization("views", df)
    df["forwards_norm"] = min_max_normalization("forwards", df)
    df["reaction_nb_norm"] = min_max_normalization("reaction_nb", df)
    df['popularity_score'] = df.apply(lambda x: weighted_popularity_score(x), axis=1)
    return df

def filter_graph(df, entities, depth=1):
    for _ in range(depth-1):
        level = df[df.source.isin(entities)] # Search only source for next nodes in path
        entities.extend(list(set(level.source.values.tolist()))) # Extend nodes search list using both ends of an edge
        entities.extend(list(set(level.target.values.tolist())))

    return df[df.source.isin(entities)]