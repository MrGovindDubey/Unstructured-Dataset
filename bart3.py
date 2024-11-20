import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import logging
from tqdm import tqdm
import nltk

# Set up logging for detailed tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download stopwords if not already available
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

# Load the dataset
input_file = r'D:\Projects\Python\train.csv'  # Update with your correct file path
output_file = r'D:\Projects\Python\bart_structured_output.csv'  # Update if needed

logging.info("Loading dataset from: %s", input_file)
df = pd.read_csv(input_file)
logging.info("Dataset loaded successfully")

# Preprocess text descriptions
logging.info("Starting text preprocessing...")
df['processed_text'] = [preprocess_text(text) for text in tqdm(df['crimeaditionalinfo'].fillna(''))]
logging.info("Text preprocessing completed")

# Split data into training and testing sets
logging.info("Splitting dataset into training and testing sets...")
df_known = df.dropna(subset=['category', 'sub_category'])
X = df_known['processed_text']
y = df_known['category'].astype('category').cat.codes  # Convert categories to integer codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Data split completed: %d training samples and %d test samples", len(X_train), len(X_test))

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=len(df['category'].unique()))

# Tokenize and prepare the dataset
def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=512)
test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": y_train.values})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": y_test.values})

# Define the Trainer and TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    evaluation_strategy="epoch",  # evaluation strategy to adopt during training
    learning_rate=2e-5,  # learning rate
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
    num_train_epochs=3,  # number of training epochs
    weight_decay=0.01,  # strength of weight decay
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
)

# Train the model
logging.info("Starting model training...")
trainer.train()

# Evaluate model performance
logging.info("Evaluating model performance...")
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

category_report = classification_report(y_test, predicted_labels)
logging.info("Category Classification Report:\n%s", category_report)

# Apply the model on the entire dataset (including missing labels) for predictions
logging.info("Applying the model to the entire dataset for predictions...")
df['predicted_category'] = df['processed_text'].apply(lambda text: model(**tokenizer(text, return_tensors="pt")).logits.argmax().item())

# Save the structured output to a new CSV file
logging.info("Saving the structured output to: %s", output_file)
df[['category', 'sub_category', 'crimeaditionalinfo', 'predicted_category']].to_csv(output_file, index=False)

logging.info("Structured data with predictions saved to %s", output_file)
print(f"Results saved to: {output_file}")
