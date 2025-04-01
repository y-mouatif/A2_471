import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt


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
