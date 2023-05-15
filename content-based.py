import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('steam_games.csv', nrows=10000)

# Select relevant features for content-based filtering
features = ['name', 'genre', 'game_details', 'popular_tags']

# Clean and preprocess the data
df['name'] = df['name'].str.lower()
df['text'] = df[features].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get top recommendations
def get_recommendations(game_title, num_recommendations=5):
    # Get the index of the game title
    idx = df[df['name'] == game_title.lower()].index[0]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the games based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of top recommendations
    top_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Return the recommended game titles
    recommendations = df['name'].iloc[top_indices].values.tolist()
    return recommendations

game_title = input("Enter game title: ")
recommended_games = get_recommendations(game_title, num_recommendations=5)
print(f"Recommended games based on '{game_title}':")
for game in recommended_games:
    print(game)
