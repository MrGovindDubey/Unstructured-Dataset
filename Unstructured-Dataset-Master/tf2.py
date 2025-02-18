import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
import nltk
import logging
from tqdm import tqdm
import time

# Set up logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download stopwords if not already available
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """Improved text preprocessing with lemmatization and additional cleaning."""
    from nltk.stem import WordNetLemmatizer

    # Lemmatizer initialization
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase, remove non-alphabetic characters, and remove stopwords
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())

    # Lemmatizing and removing stopwords
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

    return text


# Step 1: Load the dataset

input_file = r'D:\Projects\Python\train.csv'  
output_file = r'D:\Projects\Python\structured_output.csv'  


logging.info("Loading dataset from: %s", input_file)
start_time = time.time()
df = pd.read_csv(input_file)
logging.info("Dataset loaded successfully in %.2f seconds", time.time() - start_time)

# Step 2: Preprocess text descriptions
logging.info("Starting text preprocessing...")
start_time = time.time()

# Use tqdm to track the progress of the preprocessing step
df['processed_text'] = [preprocess_text(text) for text in tqdm(df['crimeaditionalinfo'].fillna(''))]

logging.info("Text preprocessing completed in %.2f seconds", time.time() - start_time)

# Step 3: Split data into training and testing sets
logging.info("Splitting dataset into training and testing sets...")
df_known = df.dropna(subset=['category', 'sub_category'])
X = df_known['processed_text']
y = df_known[['category', 'sub_category']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split completed: %d training samples and %d test samples", len(X_train), len(X_test))

# Step 4: Build a text classification pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
    # Transform text to TF-IDF features (higher max_features)
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    # Classifier with balanced class weights
])

# Step 5: Hyperparameter tuning using GridSearchCV
param_grid = {
    'clf__n_estimators': [100, 150, 200],
    'clf__max_depth': [10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
}

logging.info("Starting GridSearchCV for hyperparameter tuning...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train['category'])

logging.info("Best parameters from GridSearchCV: %s", grid_search.best_params_)

# Step 6: Train the best model with the optimal hyperparameters
best_pipeline = grid_search.best_estimator_

# Step 7: Predict categories and subcategories on the test set
logging.info("Making predictions on the test set...")

y_pred_category = best_pipeline.predict(X_test)
y_pred_sub_category = best_pipeline.predict(X_test)

# Step 8: Evaluate model performance
logging.info("Evaluating category classifier performance...")
category_report = classification_report(y_test['category'], y_pred_category)
sub_category_report = classification_report(y_test['sub_category'], y_pred_sub_category)

print("Category Classification Report:")
print(category_report)

print("Subcategory Classification Report:")
print(sub_category_report)

logging.info("Category Classification Report:\n%s", category_report)
logging.info("Subcategory Classification Report:\n%s", sub_category_report)

# Step 9: Apply the model on the entire dataset (including missing labels) for predictions
logging.info("Applying the model to the entire dataset for predictions...")
df['predicted_category'] = best_pipeline.predict(df['processed_text'])
df['predicted_sub_category'] = best_pipeline.predict(df['processed_text'])

# Step 10: Save the structured output to a new CSV file
logging.info("Saving the structured output to: %s", output_file)

# Save results to CSV, ensuring clarity in output
df[['category', 'sub_category', 'crimeaditionalinfo', 'predicted_category', 'predicted_sub_category']].to_csv(
    output_file, index=False)

logging.info("Structured data with predictions saved to %s", output_file)
print(f"Results saved to: {output_file}")
