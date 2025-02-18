from transformers import pipeline
from keybert import KeyBERT
import pandas as pd
import json

# Load the CSV file with descriptions
df = pd.read_csv('cleaned_data.csv')

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize KeyBERT model for keyword extraction
kw_model = KeyBERT()

# Ensure the column name is 'description'
descriptions = df['description']

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

# Flatten the categories for zero-shot classification
candidate_labels = [subcategory for subcategories in categories.values() for subcategory in subcategories]

# Function to classify and extract keywords
def classify_and_extract_keywords(text):
    # Classify the text into one of the candidate labels (subcategories)
    result = classifier(text, candidate_labels)
    
    # Get the predicted subcategory
    predicted_subcategory = result['labels'][0]
    
    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    extracted_keywords = [keyword for keyword, score in keywords]
    
    return predicted_subcategory, extracted_keywords

# Apply the function to each description and collect all results
all_results = []
for description in descriptions:
    predicted_subcategory, extracted_keywords = classify_and_extract_keywords(description)
    
    # Find the main category based on the predicted subcategory
    main_category = None
    for category, subcategories in categories.items():
        if predicted_subcategory in subcategories:
            main_category = category
            break
    
    # Append the result
    all_results.append({
        "description": description,
        "main_category": main_category,
        "subcategory": predicted_subcategory,
        "keywords": extracted_keywords
    })

# Save the result to a JSON file
with open('classified_keywords.json', 'w') as f:
    json.dump(all_results, f, indent=4)

print("Classified descriptions and extracted keywords saved to 'classified_keywords.json'")