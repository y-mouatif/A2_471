import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Part #1
# Loading the Data
ecommerce_data = pd.read_csv('ecommerce_user_data.csv')
product_details = pd.read_csv('product_details.csv')

# Quick inspection of the data
print(ecommerce_data.head())
print(product_details.head())

# Clean the Data
# Convert Timestamp to datetime
ecommerce_data['Timestamp'] = pd.to_datetime(ecommerce_data['Timestamp'])

# Create a User-Item Matrix
# users as rows, products as columns, ratings as values
user_item_matrix = ecommerce_data.pivot_table(index='UserID', columns='ProductID', values='Rating')

# Handle missing values by filling them with 0 //for now 
user_item_matrix_filled = user_item_matrix.fillna(0)
print(user_item_matrix_filled.head())

# 4. Merge Data for Aggregation (if needed)
# Merge ecommerce_data with product_details to ensure consistent category labels
merged_data = pd.merge(ecommerce_data, product_details, on='ProductID', suffixes=('', '_prod'))

# Group and Aggregate Purchase Behaviors
# group by UserID & Product Category (using the category from product_details)
grouped = merged_data.groupby(['UserID', 'Category']).agg({'Rating': ['count', 'mean']}).reset_index()
grouped.columns = ['UserID', 'Category', 'PurchaseCount', 'AverageRating']
print(grouped.head())

# Part #2

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix_filled)

# Put into a DataFrame for easier reading
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)
print(user_similarity_df.head())

# Get Top-N Similar Users for Each User
def get_top_n_similar_users(user_id, n=5):
    if user_id not in user_similarity_df:
        return []
    # Sort users by similarity score
    sorted_users = user_similarity_df[user_id].sort_values(ascending=False)
    # Exclude the user themself
    top_users = sorted_users.drop(user_id).head(n)
    return top_users.index.tolist()

# Recommend Products Based on Similar Users
def recommend_products(user_id, n_similar=5, n_recommendations=5):
    if user_id not in user_item_matrix_filled.index:
        return []
    
    top_users = get_top_n_similar_users(user_id, n_similar)
    
    user_ratings = user_item_matrix_filled.loc[user_id]
    unseen_products = user_ratings[user_ratings == 0].index

    # Aggregate ratings from similar users
    similar_users_ratings = user_item_matrix_filled.loc[top_users]
    avg_ratings = similar_users_ratings[unseen_products].mean(axis=0)
    
    top_products = avg_ratings.sort_values(ascending=False).head(n_recommendations)
    
    return top_products.index.tolist()

# Generate Recommendations for All Users
recommendations = {}
for user in user_item_matrix_filled.index:
    recommendations[user] = recommend_products(user, n_similar=5, n_recommendations=5)

# Convert to DataFrame
recommendations_df = pd.DataFrame.from_dict(recommendations, orient='index')
recommendations_df.columns = [f"Rec_{i+1}" for i in range(recommendations_df.shape[1])]
print(recommendations_df.head())

# Evaluate with Precision@K
def precision_at_k(user_id, recommended_items, k=5):
    actual_items = set(ecommerce_data[ecommerce_data['UserID'] == user_id]['ProductID'])
    recommended_items = set(recommended_items[:k])
    if not actual_items:
        return 0.0
    return len(actual_items.intersection(recommended_items)) / k

precision_scores = []
for user in recommendations_df.index:
    recommended = recommendations_df.loc[user].dropna().tolist()
    precision_scores.append(precision_at_k(user, recommended))

print(f"\nAverage Precision@5: {sum(precision_scores) / len(precision_scores):.2f}")

#output to csv
recommendations_df.to_csv("recommendations_output.csv", index=True)



# Part #3
# Convert into transactions
transactions = merged_data.groupby("UserID")["ProductID"].apply(list).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori and generate rules
frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# Part #4
# Heat map visualizations
plt.figure(figsize=(10, 8))
sns.heatmap(df_transactions.corr(), cmap='YlGnBu')
plt.title('User Similarity Heatmap')
plt.show()

# Frequent itemsets bar chart
frequent_itemsets.nlargest(10, 'support').plot(kind='bar', x='itemsets', y='support', legend=False)
plt.title('Top 10 Frequent Itemsets')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
