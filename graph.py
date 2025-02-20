import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import re
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 0: Check for GPU availability and configure PyTorch to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the trained model and metadata
model_save_path = "crime_cnn_model.pth"
metadata_save_path = "crime_metadata.pkl"

try:
    # Load metadata
    with open(metadata_save_path, "rb") as f:
        metadata = pickle.load(f)
    tokenizer = metadata['tokenizer']
    category_encoder = metadata['category_encoder']
    subcategory_encoder = metadata['subcategory_encoder']
    max_sequence_length = metadata['max_sequence_length']

    # Load the model
    num_categories = len(category_encoder.classes_)
    num_subcategories = len(subcategory_encoder.classes_)
    embedding_dim = 100
    max_words = 10000

    class CNNModel(nn.Module):
        def __init__(self, max_words, embedding_dim, max_sequence_length, num_categories, num_subcategories):
            super(CNNModel, self).__init__()
            self.embedding = nn.Embedding(max_words, embedding_dim)
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.global_pool = nn.AdaptiveMaxPool1d(1)
            self.dropout = nn.Dropout(0.5)
            self.fc_category = nn.Linear(128, num_categories)
            self.fc_subcategory = nn.Linear(128, num_subcategories)

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, seq_len)
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.global_pool(x).squeeze(2)
            x = self.dropout(x)
            category_output = torch.softmax(self.fc_category(x), dim=1)
            subcategory_output = torch.softmax(self.fc_subcategory(x), dim=1)
            return category_output, subcategory_output

    model = CNNModel(max_words, embedding_dim, max_sequence_length, num_categories, num_subcategories).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    logging.info("Model and metadata loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or metadata: {e}")
    exit(1)

# Load dataset for evaluation
try:
    data = pd.read_csv('crime_data.csv')  # Replace with your dataset path
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Error: Dataset file not found. Please check the file path.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    exit(1)

# Preprocess data
data.dropna(subset=['text', 'category', 'subcategory'], inplace=True)
data.drop_duplicates(inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

data['text'] = data['text'].apply(clean_text)
data['category_encoded'] = category_encoder.transform(data['category'])
data['subcategory_encoded'] = subcategory_encoder.transform(data['subcategory'])

# Tokenize and pad sequences
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=max_sequence_length)

# Convert labels
y_category = to_categorical(data['category_encoded'], num_classes=num_categories)
y_subcategory = to_categorical(data['subcategory_encoded'], num_classes=num_subcategories)

# Create PyTorch dataset and dataloader
dataset = CrimeDataset(X, y_category, y_subcategory)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Evaluate the model and generate predictions
predictions = []
model.eval()
with torch.no_grad():
    for batch_X, _, _ in loader:
        batch_X = batch_X.to(device)
        category_output, subcategory_output = model(batch_X)
        predictions.append((category_output.cpu().numpy(), subcategory_output.cpu().numpy()))

predicted_categories = np.argmax(np.vstack([p[0] for p in predictions]), axis=1)
predicted_subcategories = np.argmax(np.vstack([p[1] for p in predictions]), axis=1)

# Map predictions back to original labels
data['predicted_category'] = category_encoder.inverse_transform(predicted_categories)
data['predicted_subcategory'] = subcategory_encoder.inverse_transform(predicted_subcategories)

# Add threat scores for predictions
threat_scores = {
    # ... (same as before)
}

def get_threat_score(category, subcategory):
    return threat_scores.get(category, {}).get(subcategory, 0)

data['predicted_threat_score'] = data.apply(
    lambda row: get_threat_score(row['predicted_category'], row['predicted_subcategory']), axis=1
)

# Rank crimes by threat score
ranked_crimes = data.sort_values(by='predicted_threat_score', ascending=False)

# Save results to a CSV file
try:
    ranked_crimes.to_csv('ranked_crime_predictions.csv', index=False)
    logging.info("Ranked crime predictions saved to 'ranked_crime_predictions.csv'")
except Exception as e:
    logging.error(f"Error saving results to CSV: {e}")

# Generate Graphical Reports
plt.figure(figsize=(15, 10))

# 1. Confusion Matrix for Category Predictions
plt.subplot(2, 2, 1)
cm_category = confusion_matrix(data['category_encoded'], predicted_categories)
sns.heatmap(cm_category, annot=True, fmt='d', cmap='Blues', xticklabels=category_encoder.classes_, yticklabels=category_encoder.classes_)
plt.title("Category Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# 2. Confusion Matrix for Subcategory Predictions
plt.subplot(2, 2, 2)
cm_subcategory = confusion_matrix(data['subcategory_encoded'], predicted_subcategories)
sns.heatmap(cm_subcategory, annot=True, fmt='d', cmap='Blues', xticklabels=subcategory_encoder.classes_, yticklabels=subcategory_encoder.classes_)
plt.title("Subcategory Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# 3. Threat Score Distribution
plt.subplot(2, 2, 3)
sns.histplot(data['predicted_threat_score'], bins=20, kde=True, color='purple')
plt.title("Threat Score Distribution")
plt.xlabel("Threat Score")
plt.ylabel("Frequency")

# 4. Top 10 Ranked Crimes
plt.subplot(2, 2, 4)
top_10 = ranked_crimes.head(10)
sns.barplot(x='predicted_threat_score', y='text', data=top_10, palette='viridis')
plt.title("Top 10 Ranked Crimes by Threat Score")
plt.xlabel("Threat Score")
plt.ylabel("Crime Description")

# Save the visualization as a PDF
plt.tight_layout()
plt.savefig("crime_prediction_visualization.pdf")
plt.show()
logging.info("Graphical reports saved to 'crime_prediction_visualization.pdf'")
