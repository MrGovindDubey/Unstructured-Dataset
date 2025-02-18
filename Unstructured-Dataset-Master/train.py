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

# Check for GPU availability and configure PyTorch to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and Preprocess Data
try:
    # Assuming the dataset has columns: 'text', 'category', 'subcategory'
    data = pd.read_csv('crime_data.csv')  # Replace with your dataset path
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Encode labels
try:
    category_encoder = LabelEncoder()
    subcategory_encoder = LabelEncoder()
    data['category_encoded'] = category_encoder.fit_transform(data['category'])
    data['subcategory_encoded'] = subcategory_encoder.fit_transform(data['subcategory'])
    print("Labels encoded successfully.")
except KeyError as e:
    print(f"Error: Missing column in dataset - {e}")
    exit(1)
except Exception as e:
    print(f"Error encoding labels: {e}")
    exit(1)

# Define threat ranking (updated with all subcategories)
threat_scores = {
    'Crime against Women & Children': {
        'Rape/Gang Rape': 10,
        'Sexual Harassment': 9,
        'Cyber Stalking': 8,
        'Cyber Bullying': 7,
        'Child Pornography/Child Sexual Abuse Material (CSAM)': 10,
        'Publishing/transmitting obscene material/sexually explicit material': 6,
        'Computer-Generated CSAM/CSEM': 10,
        'Fake Social Media Profile': 5,
        'Cyber Blackmailing & Threatening': 8,
        'Online Human Trafficking': 10,
        'cyber voyeurism': 7,
        'Child sexual exploitative material (CSEM)': 10,
        'defamation': 4,
        'others ( in crime against women and children )': 3
    },
    'Financial Crimes': {
        'Investment Scam/Trading Scam': 8,
        'Online Job Fraud': 7,
        'Tech Support/Customer Care Scam': 6,
        'online loan fraud': 8,
        'Matrimonial/Romance Scam/Honey Trapping Scam': 7,
        'Impersonation of Govt. Servant': 6,
        'cheating by impresonation(other than government servant)': 5,
        'SIM Swap Fraud': 9,
        'Sextortion/Nude Video Call': 10,
        'Aadhaar-enabled Payment System (AePS) Fraud / Biometric Cloning': 9,
        'Identity Theft': 8,
        'Courier/Parcel Scam': 6,
        'Phishing': 7,
        'Online Shopping/E-commerce Frauds': 7,
        'Advance Fee Fraud': 6,
        'Real Estate/Rental Payment Fraud': 6,
        'Others ( in Financial Crimes )': 5
    },
    'Cyber Attack/Dependent Crimes': {
        'Malware Attack': 9,
        'Ransomware Attack': 10,
        'Hacking/Defacement': 9,
        'Data Breach/Theft': 9,
        'Tampering with computer source documents': 8,
        'Denial of Service (DoS)/Distributed Denial of Service (DDoS) attacks': 8,
        'SQL Injection': 8
    },
    'Other Cyber Crimes': {
        'Fake profile': 6,
        'Phishing': 7,
        'Cyber Terrorism': 10,
        'Social Media Account Hacking': 8,
        'Online Gambling/Betting Fraud': 7,
        'Business Email Compromise/Email Takeover': 8,
        'Provocative Speech for unlawful acts': 6,
        'Matrimonial/Honey Trapping Scam': 7,
        'Fake News': 5,
        'Cyber Stalking/Bullying': 7,
        'Defamation': 5,
        'Cyber Pornography': 8,
        'Sending obscene material': 6,
        'Intellectual Property (IPR) Thefts': 7,
        'Cyber Enabled Human Trafficking': 10,
        'Cyber Blackmailing & Threatening': 8,
        'Online Piracy': 6,
        'Spoofing': 6,
        'Others(Other Cyber Crime)': 5
    }
}

def get_threat_score(category, subcategory):
    return threat_scores.get(category, {}).get(subcategory, 0)

data['threat_score'] = data.apply(lambda row: get_threat_score(row['category'], row['subcategory']), axis=1)

# Split dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['category_encoded'])
train_data, val_data = train_test_split(train_data, test_size=0.125, random_state=42, stratify=train_data['category_encoded'])

# Tokenize text data
max_words = 10000  # Vocabulary size
max_sequence_length = 100  # Maximum length of sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['text'])
X_train = tokenizer.texts_to_sequences(train_data['text'])
X_val = tokenizer.texts_to_sequences(val_data['text'])
X_test = tokenizer.texts_to_sequences(test_data['text'])
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_val = pad_sequences(X_val, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# One-hot encode labels
num_categories = len(data['category_encoded'].unique())
num_subcategories = len(data['subcategory_encoded'].unique())
y_train_category = to_categorical(train_data['category_encoded'], num_classes=num_categories)
y_val_category = to_categorical(val_data['category_encoded'], num_classes=num_categories)
y_test_category = to_categorical(test_data['category_encoded'], num_classes=num_categories)
y_train_subcategory = to_categorical(train_data['subcategory_encoded'], num_classes=num_subcategories)
y_val_subcategory = to_categorical(val_data['subcategory_encoded'], num_classes=num_subcategories)
y_test_subcategory = to_categorical(test_data['subcategory_encoded'], num_classes=num_subcategories)

# Convert data to PyTorch tensors
class CrimeDataset(Dataset):
    def __init__(self, X, y_category, y_subcategory):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y_category = torch.tensor(y_category, dtype=torch.float32)
        self.y_subcategory = torch.tensor(y_subcategory, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_category[idx], self.y_subcategory[idx]

train_dataset = CrimeDataset(X_train, y_train_category, y_train_subcategory)
val_dataset = CrimeDataset(X_val, y_val_category, y_val_subcategory)
test_dataset = CrimeDataset(X_test, y_test_category, y_test_subcategory)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Build CNN Model with Joint Learning
embedding_dim = 100  # Dimension of word embeddings

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

# Compile the model
criterion_category = nn.CrossEntropyLoss()
criterion_subcategory = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Model
def train_model(model, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_X, batch_y_category, batch_y_subcategory in train_loader:
            batch_X, batch_y_category, batch_y_subcategory = batch_X.to(device), batch_y_category.to(device), batch_y_subcategory.to(device)

            optimizer.zero_grad()
            category_output, subcategory_output = model(batch_X)
            loss_category = criterion_category(category_output, batch_y_category.argmax(dim=1))
            loss_subcategory = criterion_subcategory(subcategory_output, batch_y_subcategory.argmax(dim=1))
            loss = loss_category + 0.5 * loss_subcategory
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")

train_model(model, train_loader, val_loader)

# Step 4: Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    total_correct_category = 0
    total_correct_subcategory = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y_category, batch_y_subcategory in test_loader:
            batch_X, batch_y_category, batch_y_subcategory = batch_X.to(device), batch_y_category.to(device), batch_y_subcategory.to(device)

            category_output, subcategory_output = model(batch_X)
            _, predicted_category = torch.max(category_output, 1)
            _, predicted_subcategory = torch.max(subcategory_output, 1)

            total_correct_category += (predicted_category == batch_y_category.argmax(dim=1)).sum().item()
            total_correct_subcategory += (predicted_subcategory == batch_y_subcategory.argmax(dim=1)).sum().item()
            total_samples += batch_X.size(0)

    accuracy_category = total_correct_category / total_samples
    accuracy_subcategory = total_correct_subcategory / total_samples
    print(f"Test Category Accuracy: {accuracy_category:.4f}")
    print(f"Test Subcategory Accuracy: {accuracy_subcategory:.4f}")

evaluate_model(model, test_loader)

# Step 5: Predict and Rank Crimes
predictions = []
model.eval()
with torch.no_grad():
    for batch_X, _, _ in test_loader:
        batch_X = batch_X.to(device)
        category_output, subcategory_output = model(batch_X)
        predictions.append((category_output.cpu().numpy(), subcategory_output.cpu().numpy()))

predicted_categories = np.argmax(np.vstack([p[0] for p in predictions]), axis=1)
predicted_subcategories = np.argmax(np.vstack([p[1] for p in predictions]), axis=1)

# Map predictions back to original labels
test_data['predicted_category'] = category_encoder.inverse_transform(predicted_categories)
test_data['predicted_subcategory'] = subcategory_encoder.inverse_transform(predicted_subcategories)

# Add threat scores for predictions
test_data['predicted_threat_score'] = test_data.apply(
    lambda row: get_threat_score(row['predicted_category'], row['predicted_subcategory']), axis=1
)

# Rank crimes by threat score
ranked_crimes = test_data.sort_values(by='predicted_threat_score', ascending=False)

# Save results to a CSV file
try:
    ranked_crimes.to_csv('ranked_crime_predictions.csv', index=False)
    print("Ranked crime predictions saved to 'ranked_crime_predictions.csv'")
except Exception as e:
    print(f"Error saving results to CSV: {e}")