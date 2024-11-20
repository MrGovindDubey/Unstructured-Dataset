import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase, remove non-alphabetic characters, and remove stopwords
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Step 1: Load the dataset
input_file = 'D:\\Projects\\Python\\train.csv'  
output_file = 'D:\\Projects\\Python\\structured_output.csv'

df = pd.read_csv(input_file)

# Step 2: Preprocess text descriptions
df['processed_text'] = df['crimeaditionalinfo'].fillna('').apply(preprocess_text)

# Step 3: Split data into training and testing sets
# We’ll use only the rows with known categories for training, ignoring unknowns
df_known = df.dropna(subset=['category', 'sub_category'])
X = df_known['processed_text']
y = df_known[['category', 'sub_category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),  # Transform text to TF-IDF features
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))  # Classifier
])

# Step 5: Train the model separately for 'category' and 'sub_category'
# Training Category Classifier
print("Training category classifier...")
category_pipeline = pipeline.fit(X_train, y_train['category'])

# Training Subcategory Classifier
print("Training subcategory classifier...")
sub_category_pipeline = pipeline.fit(X_train, y_train['sub_category'])

# Step 6: Predict categories and subcategories on test set
y_pred_category = category_pipeline.predict(X_test)
y_pred_sub_category = sub_category_pipeline.predict(X_test)

# Step 7: Evaluate model performance
print("Category Classification Report:")
print(classification_report(y_test['category'], y_pred_category))
print("Subcategory Classification Report:")
print(classification_report(y_test['sub_category'], y_pred_sub_category))

# Step 8: Apply the model on the entire dataset (including missing labels) for predictions
df['predicted_category'] = category_pipeline.predict(df['processed_text'])
df['predicted_sub_category'] = sub_category_pipeline.predict(df['processed_text'])

# Step 9: Save the structured output to a new CSV file
df[['category', 'sub_category', 'crimeaditionalinfo', 'predicted_category', 'predicted_sub_category']].to_csv(output_file, index=False)
print(f"Structured data with predictions saved to {output_file}")
