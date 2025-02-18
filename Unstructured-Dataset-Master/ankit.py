import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Preprocess Data
# Assuming the dataset has columns: 'text', 'category', 'subcategory'
data = pd.read_csv('crime_data.csv')  # Replace with your dataset path

# Encode labels
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

data['category_encoded'] = category_encoder.fit_transform(data['category'])
data['subcategory_encoded'] = subcategory_encoder.fit_transform(data['subcategory'])

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

# Step 2: Build CNN Model with Joint Learning
embedding_dim = 100  # Dimension of word embeddings

def build_model(max_words, embedding_dim, max_sequence_length, num_categories, num_subcategories):
    inputs = Input(shape=(max_sequence_length,))
    
    # Embedding layer
    x = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
    
    # Shared CNN layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    
    # Output layers
    category_output = Dense(num_categories, activation='softmax', name='category_output')(x)
    subcategory_output = Dense(num_subcategories, activation='softmax', name='subcategory_output')(x)
    
    model = Model(inputs=inputs, outputs=[category_output, subcategory_output])
    return model

model = build_model(max_words, embedding_dim, max_sequence_length, num_categories, num_subcategories)

# Compile the model
model.compile(optimizer='adam',
              loss={'category_output': 'categorical_crossentropy',
                    'subcategory_output': 'categorical_crossentropy'},
              loss_weights={'category_output': 1.0, 'subcategory_output': 0.5},
              metrics=['accuracy'])

# Step 3: Train the Model
history = model.fit(X_train,
                    {'category_output': y_train_category, 'subcategory_output': y_train_subcategory},
                    validation_data=(X_val, 
                                     {'category_output': y_val_category, 'subcategory_output': y_val_subcategory}),
                    epochs=10, batch_size=32)

# Step 4: Evaluate the Model
test_loss, test_category_loss, test_subcategory_loss, test_category_acc, test_subcategory_acc = model.evaluate(
    X_test,
    {'category_output': y_test_category, 'subcategory_output': y_test_subcategory}
)

print(f"Test Category Accuracy: {test_category_acc}")
print(f"Test Subcategory Accuracy: {test_subcategory_acc}")

# Step 5: Predict and Rank Crimes
predictions = model.predict(X_test)
predicted_categories = np.argmax(predictions[0], axis=1)
predicted_subcategories = np.argmax(predictions[1], axis=1)

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
ranked_crimes.to_csv('ranked_crime_predictions.csv', index=False)

print("Ranked crime predictions saved to 'ranked_crime_predictions.csv'")