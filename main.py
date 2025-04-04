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

# Calculating the sparsity of user-item matrix
num_users, num_items = user_item_matrix.shape
total_possible = num_users * num_items
actual_ratings = user_item_matrix.count().sum()
sparsity = 1 - (actual_ratings / total_possible)

print(f"Sparsity of user-item matrix: {sparsity:.2%}")


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
def get_top_n_similar_users(user_id, n=5, threshold=0.1):
    if user_id not in user_similarity_df:
        return []
    sorted_users = user_similarity_df[user_id].sort_values(ascending=False)
    # Filter out weak similarities and the user themself
    filtered = sorted_users[(sorted_users > threshold) & (sorted_users.index != user_id)]
    return filtered.head(n).index.tolist()


# Function to recommend products for a given user based on similar users
def recommend_products(user_id, n_similar=5, n_recommendations=5):
    # Skip if user not found in matrix
    if user_id not in user_item_matrix_filled.index:
        return []
    
    # Get top N similar users with similarity threshold applied
    top_users = get_top_n_similar_users(user_id, n_similar)
    if not top_users:
        return []
    
    # Identify which products the current user hasn't rated yet
    user_ratings = user_item_matrix_filled.loc[user_id]
    unseen_products = user_ratings[user_ratings == 0].index

    # Get ratings from similar users for only unseen products
    similar_users_ratings = user_item_matrix_filled.loc[top_users]
    
    # Only consider ratings that are 3 or higher (positive feedback)
    positive_ratings = similar_users_ratings[unseen_products]
    positive_ratings = positive_ratings.where(positive_ratings >= 3)

    # Average the ratings from similar users and drop products with no ratings
    avg_ratings = positive_ratings.mean(axis=0).dropna()
    if avg_ratings.empty:
        return []
    
     # Return top-N products with highest average rating
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

# Create a mapping from ProductID to Category
product_to_category = product_details.set_index('ProductID')['Category'].to_dict()

# Precision@K: How many of the recommended categories match the user’s actual preferred categories
def precision_at_k(user_id, recommended_items, k=5):
    actual_positive_categories = set(
        ecommerce_data[
            (ecommerce_data['UserID'] == user_id) & 
            (ecommerce_data['Rating'] >= 3)
        ]['ProductID'].map(product_to_category)
    )

    recommended_categories = set(pd.Series(recommended_items[:k]).map(product_to_category))

    if not actual_positive_categories:
        return 0.0

    return len(actual_positive_categories.intersection(recommended_categories)) / k

# Recall@K: How many of the user's actual preferred categories are captured in the recommendations
def recall_at_k(user_id, recommended_items, k=5):
    actual_positive_categories = set(
        ecommerce_data[
            (ecommerce_data['UserID'] == user_id) & 
            (ecommerce_data['Rating'] >= 3)
        ]['ProductID'].map(product_to_category)
    )

    recommended_categories = set(pd.Series(recommended_items[:k]).map(product_to_category))

    if not actual_positive_categories:
        return 0.0

    return len(actual_positive_categories.intersection(recommended_categories)) / len(actual_positive_categories)

# Collect precision and recall scores across all users
precision_scores = []
recall_scores = []

# Compute average precision and recall for the entire system
for user in recommendations_df.index:
    recommended = recommendations_df.loc[user].dropna().tolist()
    precision_scores.append(precision_at_k(user, recommended))
    recall_scores.append(recall_at_k(user, recommended))

avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)

print(f"Average Precision@5: {avg_precision:.2f}")
print(f"Average Recall@5: {avg_recall:.2f}")

# Calculate coverage: % of users who received at least 1 recommendation
users_with_recommendations = recommendations_df.apply(lambda row: row.notna().any(), axis=1).sum()
total_users = len(recommendations_df)
coverage = users_with_recommendations / total_users

print(f"Recommendation Coverage: {coverage:.2%}")

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
