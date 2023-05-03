import pandas as pd
import numpy as np
import re
from collections import Counter

df = pd.read_csv("steam_games.csv", nrows=100)

def preprocess(text):
    text = text.lower() 
    text = re.sub(r"[^a-z0-9]+", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()  
    return text

df["clean_description"] = df["desc_snippet"].fillna("") + " " + df["game_description"].fillna("")
df["clean_description"] = df["clean_description"].apply(preprocess)


doc_freqs = Counter()
for desc in df["clean_description"]:
    words = set(desc.split())
    for word in words:
        doc_freqs[word] += 1

tf_idf_matrix = np.zeros((len(df), len(doc_freqs)))
for i, desc in enumerate(df["clean_description"]):
    word_counts = Counter(desc.split())
    for j, word in enumerate(doc_freqs.keys()):
        tf = word_counts[word] / len(desc.split())
        idf = np.log(len(df) / doc_freqs[word])
        tf_idf_matrix[i, j] = tf * idf

def recommend_games(game_name, n=10):
    if game_name not in df["name"].values:
        return "Invalid game name"
    game_index = df[df["name"] == game_name].index[0]
    scores = tf_idf_matrix.dot(tf_idf_matrix[game_index])
    top_indices = scores.argsort()[::-1][1:n+1]
    return list(df.iloc[top_indices]["name"])


game = input("\nEnter the game name: ")
print(recommend_games(str(game), n=5))