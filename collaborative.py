import pandas as pd

# Read in the Steam dataset
steam_df = pd.read_csv('steam-200k.csv', header=None, names=['user_id', 'game_title', 'behavior', 'hours', 'other'])

# Keep only the columns we need
steam_df = steam_df[['user_id', 'game_title', 'behavior', 'hours']]

# Keep only rows where the user has played the game for more than 10 hours
steam_df = steam_df[steam_df['behavior'] == 'play']
steam_df = steam_df[steam_df['hours'] > 10]

# Create a pivot table where rows are user_id and columns are game_title
steam_pivot = steam_df.pivot_table(index='user_id', columns='game_title', values='hours')

# Calculate the correlation between each pair of games
game_correlation = steam_pivot.corr()

def recommend_games(user_id):
    """
    Recommends games for the given user_id
    """
    # Find all the games the user has played
    user_games = steam_pivot.loc[user_id].dropna().index

    # Calculate the correlation between each game the user has played and every other game
    correlations = pd.DataFrame()
    for game in user_games:
        similar_games = game_correlation[game].dropna()
        similar_games = similar_games.apply(lambda x: x * steam_pivot.loc[user_id, game])
        correlations = pd.concat([correlations, similar_games], axis=1)

    # Sum up the correlations for each game
    summed_correlations = correlations.sum(axis=1)

    # Remove games the user has already played
    recommendations = summed_correlations.drop(user_games, errors='ignore')

    # Sort the recommendations by descending order of correlation
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations

# Test the recommendation system
recommendations = recommend_games(59945701)
print(recommendations.head(10))
