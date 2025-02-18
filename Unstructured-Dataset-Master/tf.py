import os
import torch
import pandas as pd
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Ensure correct GPU setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CSV file with descriptions
df = pd.read_csv('cleaned_data.csv')

# Ensure the column name is 'description' and handle missing values
descriptions = df['description'].fillna("")

# Define the four main categories and their subcategories
categories = {
    "Financial Crimes": [
        "Investment Scam/Trading Scam",
        "Online Job Fraud",
        "Tech Support/Customer Care Scam",
        "Online Loan Fraud",
        "Matrimonial/Romance Scam/Honey Trapping Scam",
        "Impersonation of Govt. Servant",
        "Cheating by Impersonation (Other than Government Servant)",
        "SIM Swap Fraud",
        "Sextortion/Nude Video Call",
        "Aadhaar-enabled Payment System (AePS) Fraud / Biometric Cloning",
        "Identity Theft",
        "Courier/Parcel Scam",
        "Phishing",
        "Online Shopping/E-commerce Frauds",
        "Advance Fee Fraud",
        "Real Estate/Rental Payment Fraud",
        "Others (in Financial Crimes)"
    ],
    "Crime Against Women & Children": [
        "Rape/Gang Rape",
        "Sexual Harassment",
        "Cyber Stalking",
        "Cyber Bullying",
        "Child Pornography/Child Sexual Abuse Material (CSAM)",
        "Publishing/Transmitting Obscene Material/Sexually Explicit Material",
        "Computer-Generated CSAM/CSEM",
        "Fake Social Media Profile",
        "Cyber Blackmailing & Threatening",
        "Online Human Trafficking",
        "Cyber Voyeurism",
        "Defamation",
        "Others (in Crime Against Women & Children)"
    ],
    "Cyber Attack/Dependent Crimes": [
        "Malware Attack",
        "Ransomware Attack",
        "Hacking/Defacement",
        "Data Breach/Theft",
        "Tampering with Computer Source Documents",
        "Denial of Service (DoS)/Distributed Denial of Service (DDoS) Attacks"
    ],
    "Other Cyber Crimes": [
        "Fake Profile",
        "Phishing",
        "Cyber Terrorism",
        "Social Media Account Hacking",
        "Online Gambling/Betting Fraud",
        "Business Email Compromise/Email Takeover",
        "Provocative Speech for Unlawful Acts",
        "Matrimonial/Honey Trapping Scam",
        "Fake News",
        "Cyber Stalking/Bullying",
        "Defamation",
        "Cyber Pornography",
        "Sending Obscene Material",
        "Intellectual Property (IPR) Thefts",
        "Cyber Enabled Human Trafficking / Cyber Slavery",
        "Cyber Blackmailing & Threatening",
        "Online Piracy",
        "Spoofing",
        "Others"
    ]
}

# Flatten the categories for classification
candidate_labels = [subcategory for subcategories in categories.values() for subcategory in subcategories]

# Load the paraphrase-MiniLM-L3-v2 model
print("Loading the paraphrase-MiniLM-L3-v2 model...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=DEVICE)

# Encode candidate labels once
print("Encoding candidate labels...")
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)

# Initialize TF-IDF Vectorizer
print("Fitting TF-IDF vectorizer on the dataset...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf_vectorizer.fit(descriptions)

# Function to classify and extract keywords
def classify_and_extract_keywords(text):
    try:
        # Encode the input text
        text_embedding = model.encode(text, convert_to_tensor=True)

        # Compute cosine similarity between text and candidate labels
        similarity_scores = util.cos_sim(text_embedding, label_embeddings)
        predicted_idx = similarity_scores.argmax().item()
        predicted_subcategory = candidate_labels[predicted_idx]

        # Extract keywords using TF-IDF
        tfidf_scores = tfidf_vectorizer.transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        sorted_indices = tfidf_scores.toarray()[0].argsort()[::-1]
        extracted_keywords = [feature_names[idx] for idx in sorted_indices[:10]]

        return predicted_subcategory, extracted_keywords
    except Exception as e:
        print(f"Error processing description: {text[:50]}... Error: {str(e)}")
        return None, []

# Process descriptions in batches
batch_size = 1000  # Adjust batch size based on memory
total_descriptions = len(descriptions)
all_results = []

print(f"Starting classification and keyword extraction for {total_descriptions} descriptions...\n")

start_time = time.time()

for start_idx in range(0, total_descriptions, batch_size):
    batch = descriptions[start_idx:start_idx + batch_size]
    batch_results = []

    for description in batch:
        predicted_subcategory, extracted_keywords = classify_and_extract_keywords(description)
        
        # Find the main category based on the predicted subcategory
        main_category = next((cat for cat, subs in categories.items() if predicted_subcategory in subs), None)
        
        batch_results.append({
            "description": description,
            "main_category": main_category,
            "subcategory": predicted_subcategory,
            "keywords": extracted_keywords
        })
    
    # Append batch results to the main list
    all_results.extend(batch_results)
    
    # Save intermediate results to avoid data loss
    with open(f'classified_keywords_checkpoint_{start_idx}.json', 'w') as f:
        json.dump(batch_results, f, indent=4)
    
    elapsed_time = time.time() - start_time
    print(f"Processed {start_idx + len(batch)}/{total_descriptions} descriptions... Elapsed time: {elapsed_time:.2f} seconds")
    torch.cuda.empty_cache()  # Free GPU memory after each batch

# Save the final result
with open('classified_keywords.json', 'w') as f:
    json.dump(all_results, f, indent=4)

end_time = time.time()
total_time = end_time - start_time

print("\nClassification and keyword extraction completed!")
print(f"Total time taken: {total_time:.2f} seconds")
print("Classified descriptions and extracted keywords saved to 'classified_keywords.json'")
