import pandas as pd
import math

# data sanitation
df = pd.read_csv('steam-200k.csv', header=None, names=['UserID', 'Game', 'Behavior', 'Hours', 'Empty'],nrows=100)
df = df.drop('Empty', axis=1)
df = df[df['Behavior'] == 'play']
df['Behavior'] = df['Behavior'].str.strip()


# the correlation function used to find similar users
def pearson_correlation(user1, user2):
    user1_games = df[df['UserID'] == user1]
    user2_games = df[df['UserID'] == user2]
    common_games = pd.merge(user1_games, user2_games, on='Game', how='inner')
    if len(common_games) == 0:
        return 0
    sum1 = common_games['Hours_x'].sum()
    sum2 = common_games['Hours_y'].sum()
    sum1_sq = (common_games['Hours_x'] ** 2).sum()
    sum2_sq = (common_games['Hours_y'] ** 2).sum()
    product_sum = (common_games['Hours_x'] * common_games['Hours_y']).sum()
    num = product_sum - (sum1 * sum2 / len(common_games))
    den = math.sqrt((sum1_sq - (sum1 ** 2) / len(common_games)) * (sum2_sq - (sum2 ** 2) / len(common_games)))
    if den == 0:
        return 0
    correlation = num / den
    return correlation


def predict_rating(user, game, k):
    similar_users = {}
    user_games = df[df['UserID'] == user]
    for index, row in user_games.iterrows():
        other_users = df[df['Game'] == row['Game']]
        for index2, row2 in other_users.iterrows():
            if row2['UserID'] != user:
                similarity = pearson_correlation(user, row2['UserID'])
                if similarity > 0:
                    if row2['UserID'] in similar_users:
                        similar_users[row2['UserID']] += similarity
                    else:
                        similar_users[row2['UserID']] = similarity

    similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)
    similar_users = similar_users[:k]
    weighted_sum = 0
    total_similarity = 0
    for similar_user, similarity in similar_users:
        user_ratings = df[(df['UserID'] == similar_user) & (df['Game'] == game)]
        if len(user_ratings) > 0:
            rating = user_ratings.iloc[0]['Hours']
            weighted_sum += rating * similarity
            total_similarity += similarity
    if total_similarity == 0:
        return 0
    predicted_rating = weighted_sum / total_similarity
    return predicted_rating


def recommend_games(user, k, num_recommendations=5):
    game_list = df['Game'].unique()
    predicted_ratings = []
    for game in game_list:
        predicted_rating = predict_rating(user, game, k)
        predicted_ratings.append((game, predicted_rating))
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)
    recommended_games = [game for game, _ in predicted_ratings[:num_recommendations]]
    return recommended_games


# Example usage
user = input("Please enter user id:")  # User ID
k = 5  # Number of neighbors to consider

recommended_games = recommend_games(int(user), int(k))
print(f"Recommended games for user {user}:")
for game in recommended_games:
    print(game)
