import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Step 1: Define your predefined categories and subcategories
main_categories = [
    'Crime Against Women & Children',
    'Financial Crimes',
    'Cyber Crimes',
    'Other Crimes'
]

subcategories = {
    # Crime Against Women & Children
    'Rape / Gang Rape': 'Crime Against Women & Children',
    'Sexual Harassment': 'Crime Against Women & Children',
    'Voyeurism': 'Crime Against Women & Children',
    'Cyber Stalking': 'Crime Against Women & Children',
    'Cyber Bullying': 'Crime Against Women & Children',
    'Child Pornography/CSAM': 'Crime Against Women & Children',
    'Publishing Obscene Material': 'Crime Against Women & Children',
    'Fake Social Media Profile': 'Crime Against Women & Children',
    'Cyber Blackmailing': 'Crime Against Women & Children',
    'Online Human Trafficking': 'Crime Against Women & Children',
    
    # Financial Crimes
    'Investment Scam': 'Financial Crimes',
    'Online Job Fraud': 'Financial Crimes',
    'Tech Support Scam': 'Financial Crimes',
    'Online Financial Fraud': 'Financial Crimes',
    'Matrimonial Scam': 'Financial Crimes',
    'Impersonation of Govt. Servant': 'Financial Crimes',
    'SIM Swap Fraud': 'Financial Crimes',
    'Sextortion': 'Financial Crimes',
    'Aadhaar Fraud': 'Financial Crimes',
    'Identity Theft': 'Financial Crimes',
    'Courier Scam': 'Financial Crimes',
    'Phishing': 'Financial Crimes',
    'E-commerce Fraud': 'Financial Crimes',
    'Advance Fee Fraud': 'Financial Crimes',
    'Real Estate Fraud': 'Financial Crimes',
    
    # Add remaining subcategories here...
}

# Step 2: Load the unstructured dataset
unstructured_data = pd.read_csv(r'C:\Users\Asus\Documents\un.csv')  # Replace with your file path

# Debugging: Print column names and first few rows
print("Column Names:", unstructured_data.columns)
print("First Few Rows:")
print(unstructured_data.head())

# Ensure the column name matches your dataset
description_column = 'crime aditional information'  # Updated column name
if description_column not in unstructured_data.columns:
    raise ValueError(f"Column '{description_column}' not found in dataset. Available columns: {unstructured_data.columns.tolist()}")

# Step 3: Preprocess the description column
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    return text

# Apply preprocessing
unstructured_data['cleaned_description'] = unstructured_data[description_column].apply(preprocess_text)

# Step 4: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')  # Limit to top 5000 features
X = tfidf.fit_transform(unstructured_data['cleaned_description'])

# Step 5: Apply Clustering (K-Means)
num_clusters = len(subcategories)  # Number of clusters equals the number of subcategories
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
unstructured_data['Cluster'] = kmeans.fit_predict(X)

# Step 6: Analyze Clusters and Map to Subcategories
# Get the most frequent words in each cluster to help assign categories
def get_top_keywords_for_cluster(tfidf_matrix, feature_names, cluster_labels, top_n=5):
    cluster_keywords = {}
    for cluster in range(num_clusters):
        cluster_indices = cluster_labels == cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].sum(axis=0)
        top_word_indices = cluster_tfidf.argsort()[0, -top_n:][::-1]
        cluster_keywords[cluster] = [feature_names[i] for i in top_word_indices]
    return cluster_keywords

feature_names = tfidf.get_feature_names_out()
cluster_keywords = get_top_keywords_for_cluster(X, feature_names, unstructured_data['Cluster'])

# Print cluster keywords for manual mapping
print("Cluster Keywords:")
for cluster, keywords in cluster_keywords.items():
    print(f"Cluster {cluster}: {', '.join(keywords)}")

# Step 7: Manually Map Clusters to Subcategories
# After reviewing the cluster keywords, create a mapping from clusters to subcategories
cluster_to_subcategory = {
    0: 'Rape / Gang Rape',
    1: 'Sexual Harassment',
    2: 'Cyber Stalking',
    # Add mappings for all clusters...
}

# Map clusters to subcategories and main categories
unstructured_data['Predicted Subcategory'] = unstructured_data['Cluster'].map(cluster_to_subcategory)
unstructured_data['Predicted Main Category'] = unstructured_data['Predicted Subcategory'].map(subcategories)

# Step 8: Save the structured CSV
output_file = 'structured_data.csv'
unstructured_data.to_csv(output_file, index=False)

print(f"\nStructured CSV saved successfully to {output_file}")