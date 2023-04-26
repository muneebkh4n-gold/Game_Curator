import pandas as pd

steam_data = pd.read_csv('steam-200k.csv', header=None, names=['user_id', 'game_title', 'purchase_play', 'hours_played', 'other'])

steam_data = steam_data.drop(['purchase_play', 'other'], axis=1)

def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

similarity_dict = {}
for title1 in steam_data['game_title'].unique():
    for title2 in steam_data['game_title'].unique():
        if title1 != title2:
            similarity_dict[(title1, title2)] = jaccard_similarity(title1, title2)

def recommend_games(user_id):
    user_data = steam_data[steam_data['user_id'] == user_id]
    user_games = set(user_data['game_title'].unique())

    scores = {}
    for game in steam_data['game_title'].unique():
        if game not in user_games:
            similarity_sum = 0
            weighted_sum = 0
            for played_game in user_games:
                similarity = similarity_dict.get((game, played_game), 0)
                similarity_sum += similarity
                weighted_sum += similarity * user_data[user_data['game_title'] == played_game]['hours_played'].iloc[0]
            scores[game] = weighted_sum / similarity_sum if similarity_sum > 0 else 0

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]


print(recommend_games(59945701))