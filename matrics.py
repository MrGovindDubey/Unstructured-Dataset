import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 0: Load Model and Metadata
def load_model_and_metadata(model_path, metadata_path):
    try:
        # Load model
        model = CNNModel(max_words=10000, embedding_dim=100, max_sequence_length=100, num_categories=4, num_subcategories=50)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        logging.info(f"Model loaded successfully from {model_path}")

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        logging.info(f"Metadata loaded successfully from {metadata_path}")
        return model, metadata
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading model/metadata: {e}")
        exit(1)

# Step 1: Evaluate Model and Generate Metrics
def evaluate_model_with_metrics(model, test_loader, category_encoder, subcategory_encoder):
    model.eval()
    y_true_category, y_pred_category = [], []
    y_true_subcategory, y_pred_subcategory = []

    with torch.no_grad():
        for batch_X, batch_y_category, batch_y_subcategory in test_loader:
            batch_X, batch_y_category, batch_y_subcategory = (
                batch_X.to(device),
                batch_y_category.to(device),
                batch_y_subcategory.to(device),
            )

            category_output, subcategory_output = model(batch_X)
            _, predicted_category = torch.max(category_output, 1)
            _, predicted_subcategory = torch.max(subcategory_output, 1)

            y_true_category.extend(batch_y_category.argmax(dim=1).cpu().numpy())
            y_pred_category.extend(predicted_category.cpu().numpy())
            y_true_subcategory.extend(batch_y_subcategory.argmax(dim=1).cpu().numpy())
            y_pred_subcategory.extend(predicted_subcategory.cpu().numpy())

    # Calculate metrics
    def calculate_metrics(y_true, y_pred, label_encoder, label_type):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        logging.info(f"\n--- {label_type.capitalize()} Metrics ---")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

        # Log classification report
        report = classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
        logging.info(f"\nClassification Report:\n{report}")

    calculate_metrics(y_true_category, y_pred_category, category_encoder, "Category")
    calculate_metrics(y_true_subcategory, y_pred_subcategory, subcategory_encoder, "Subcategory")

# Step 2: Rank Crimes and Display Top Results
def rank_crimes_and_display_results(test_data, category_encoder, subcategory_encoder):
    # Map predictions back to original labels
    test_data['predicted_category'] = category_encoder.inverse_transform(test_data['predicted_category_encoded'])
    test_data['predicted_subcategory'] = subcategory_encoder.inverse_transform(test_data['predicted_subcategory_encoded'])

    # Add threat scores for predictions
    test_data['predicted_threat_score'] = test_data.apply(
        lambda row: get_threat_score(row['predicted_category'], row['predicted_subcategory']), axis=1
    )

    # Rank crimes by threat score
    ranked_crimes = test_data.sort_values(by='predicted_threat_score', ascending=False)

    # Display top 10 ranked crimes
    logging.info("\n--- Top 10 Ranked Crimes ---")
    top_ranked = ranked_crimes.head(10)[['text', 'predicted_category', 'predicted_subcategory', 'predicted_threat_score']]
    logging.info(top_ranked.to_string(index=False))

# Main Execution
if __name__ == "__main__":
    # Paths to model and metadata
    model_path = "crime_cnn_model.pth"
    metadata_path = "crime_metadata.pkl"

    # Load model and metadata
    model, metadata = load_model_and_metadata(model_path, metadata_path)

    # Extract metadata
    tokenizer = metadata['tokenizer']
    category_encoder = metadata['category_encoder']
    subcategory_encoder = metadata['subcategory_encoder']
    max_sequence_length = metadata['max_sequence_length']

    # Load test data (replace with your dataset path)
    try:
        test_data = pd.read_csv('crime_data.csv')  # Replace with your dataset path
        logging.info("Test dataset loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading test dataset: {e}")
        exit(1)

    # Preprocess test data
    X_test = tokenizer.texts_to_sequences(test_data['text'])
    X_test = pad_sequences(X_test, maxlen=max_sequence_length)
    y_test_category = to_categorical(test_data['category_encoded'], num_classes=len(category_encoder.classes_))
    y_test_subcategory = to_categorical(test_data['subcategory_encoded'], num_classes=len(subcategory_encoder.classes_))

    # Create DataLoader
    test_dataset = CrimeDataset(X_test, y_test_category, y_test_subcategory)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate model and display metrics
    evaluate_model_with_metrics(model, test_loader, category_encoder, subcategory_encoder)

    # Rank crimes and display top results
    test_data['predicted_category_encoded'] = np.argmax(y_test_category, axis=1)
    test_data['predicted_subcategory_encoded'] = np.argmax(y_test_subcategory, axis=1)
    rank_crimes_and_display_results(test_data, category_encoder, subcategory_encoder)
