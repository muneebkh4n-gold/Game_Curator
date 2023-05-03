import pandas as pd

steam_df = pd.read_csv('steam-200k.csv', header=None, names=['user_id', 'game_title', 'behavior', 'hours', 'other'])

steam_df = steam_df[['user_id', 'game_title', 'behavior', 'hours']]

steam_df = steam_df[steam_df['behavior'] == 'play']
steam_df = steam_df[steam_df['hours'] > 10]

steam_pivot = steam_df.pivot_table(index='user_id', columns='game_title', values='hours')

game_correlation = steam_pivot.corr()

def recommend_games(user_id):
    user_games = steam_pivot.loc[user_id].dropna().index

    correlations = pd.DataFrame()
    for game in user_games:
        similar_games = game_correlation[game].dropna()
        similar_games = similar_games.apply(lambda x: x * steam_pivot.loc[user_id, game])
        correlations = pd.concat([correlations, similar_games], axis=1)

    summed_correlations = correlations.sum(axis=1)

    recommendations = summed_correlations.drop(user_games, errors='ignore')

    recommendations = recommendations.sort_values(ascending=False)

    return recommendations

user_id = input("Enter user id: ")
recommendations = recommend_games(int(user_id))
print(recommendations.head(10))
