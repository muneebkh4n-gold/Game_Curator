import pandas as pd
import math

# Load the dataset
df = pd.read_csv('steam-200k.csv', header=None, names=['UserID', 'Game', 'Behavior', 'Hours', 'Empty'], nrows=100)

# Preprocess the dataset
df = df.drop('Empty', axis=1)


# Sanitize the dataset
df['Behavior'] = df['Behavior'].str.strip()  # Strip leading/trailing whitespace

# Calculate similarity between games using Pearson correlation
def pearson_correlation(game1, game2):
    game1_users = df[df['Game'] == game1]
    game2_users = df[df['Game'] == game2]
    common_users = pd.merge(game1_users, game2_users, on='UserID', how='inner')
    
    if len(common_users) == 0:
        return 0
    
    sum1 = common_users['Hours_x'].sum()
    sum2 = common_users['Hours_y'].sum()
    
    sum1_sq = (common_users['Hours_x'] ** 2).sum()
    sum2_sq = (common_users['Hours_y'] ** 2).sum()
    
    product_sum = (common_users['Hours_x'] * common_users['Hours_y']).sum()
    
    num = product_sum - (sum1 * sum2 / len(common_users))
    den = math.sqrt((sum1_sq - (sum1 ** 2) / len(common_users)) * (sum2_sq - (sum2 ** 2) / len(common_users)))
    
    if den == 0:
        return 0
    
    correlation = num / den
    return correlation

# Predict the rating using mean-based prediction function
def predict_rating(user, game, k):
    similar_games = {}
    user_games = df[df['UserID'] == user]
    
    for index, row in user_games.iterrows():
        other_games = df[df['UserID'] != user]
        
        for index2, row2 in other_games.iterrows():
            similarity = pearson_correlation(row['Game'], row2['Game'])
            
            if similarity > 0:
                if row2['Game'] in similar_games:
                    similar_games[row2['Game']] += similarity
                else:
                    similar_games[row2['Game']] = similarity
    
    similar_games = sorted(similar_games.items(), key=lambda x: x[1], reverse=True)
    similar_games = similar_games[:k]
    
    weighted_sum = 0
    total_similarity = 0
    
    for similar_game, similarity in similar_games:
        user_ratings = df[(df['UserID'] == user) & (df['Game'] == similar_game)]
        
        if len(user_ratings) > 0:
            rating = user_ratings.iloc[0]['Hours']
            weighted_sum += rating * similarity
            total_similarity += similarity
    
    if total_similarity == 0:
        return 0
    
    predicted_rating = weighted_sum / total_similarity
    return predicted_rating

# Recommend games for a particular user
def recommend_games(user, num_recommendations=5):
    user_games = df[df['UserID'] == user]['Game'].unique()
    all_games = df['Game'].unique()
    recommended_games = []
    
    for game in all_games:
        if game not in user_games:
            predicted_rating = predict_rating(user, game, k=5)
            recommended_games.append((game, predicted_rating))
    recommended_games = sorted(recommended_games, key=lambda x: x[1], reverse=True)
    recommended_games = recommended_games[:num_recommendations]
    return recommended_games


user = input("Please enter user ID: ")
num_recommendations = 5
recommended_games = recommend_games(int(user), num_recommendations)
print(f"Recommended games for user {user}:")
for game, predicted_rating in recommended_games:
    print(f"- {game}")
