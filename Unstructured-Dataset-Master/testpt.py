import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Load the CSV file
input_file = "cleaned_data.csv"
output_file = "categorized_cybercrime.csv"

try:
    data = pd.read_csv(input_file)
    if "description" not in data.columns:
        raise ValueError("Input CSV must contain a 'description' column.")
except Exception as e:
    logging.error(f"Error loading input file: {e}")
    exit(1)

# Step 2: Define predefined categories and subcategories
categories = {
    # Cyber Attack/Dependent Crimes
    "Malware Attack": {
        "keywords": ["malware", "virus", "trojan"],
        "subcategories": {}
    },
    "Ransomware Attack": {
        "keywords": ["ransomware", "encrypt files"],
        "subcategories": {}
    },
    "Hacking/Defacement": {
        "keywords": ["hack", "deface", "breach"],
        "subcategories": {}
    },
    "Data Breach/Theft": {
        "keywords": ["data breach", "data theft", "leak"],
        "subcategories": {}
    },
    "Tampering with computer source documents": {
        "keywords": ["tamper", "source code", "modify"],
        "subcategories": {}
    },
    "Denial of Service (DoS)/Distributed Denial of Service (DDoS) attacks": {
        "keywords": ["dos", "ddos", "service attack"],
        "subcategories": {}
    },

    # Other Cyber Crimes
    "Fake profile": {
        "keywords": ["fake profile", "fraudulent account"],
        "subcategories": {}
    },
    "Phishing": {
        "keywords": ["phish", "email fraud", "spoof"],
        "subcategories": {}
    },
    "Cyber Terrorism": {
        "keywords": ["cyber terror", "terrorist"],
        "subcategories": {}
    },
    "Social Media Account Hacking": {
        "keywords": ["social media hack", "account takeover"],
        "subcategories": {}
    },
    "Online Gambling/Betting Fraud": {
        "keywords": ["gambling", "betting fraud"],
        "subcategories": {}
    },
    "Business Email Compromise/Email Takeover": {
        "keywords": ["email compromise", "email takeover"],
        "subcategories": {}
    },
    "Provocative Speech for unlawful acts": {
        "keywords": ["provocative speech", "unlawful act"],
        "subcategories": {}
    },
    "Matrimonial/Honey Trapping Scam": {
        "keywords": ["matrimonial scam", "honey trap"],
        "subcategories": {}
    },
    "Fake News": {
        "keywords": ["fake news", "false information"],
        "subcategories": {}
    },
    "Cyber Stalking/Bullying": {
        "keywords": ["cyber stalk", "cyber bully"],
        "subcategories": {}
    },
    "Defamation": {
        "keywords": ["defame", "character assassination"],
        "subcategories": {}
    },
    "Cyber Pornography": {
        "keywords": ["cyber porn", "online pornography"],
        "subcategories": {}
    },
    "Sending obscene material": {
        "keywords": ["obscene", "sexually explicit"],
        "subcategories": {}
    },
    "Intellectual Property (IPR) Thefts": {
        "keywords": ["ipr theft", "copyright violation"],
        "subcategories": {}
    },
    "Cyber Enabled Human Trafficking": {
        "keywords": ["human trafficking", "cyber trafficking"],
        "subcategories": {}
    },
    "Cyber Blackmailing & Threatening": {
        "keywords": ["blackmail", "threaten"],
        "subcategories": {}
    },
    "Online Piracy": {
        "keywords": ["piracy", "illegal download"],
        "subcategories": {}
    },
    "Spoofing": {
        "keywords": ["spoof", "fake identity"],
        "subcategories": {}
    },
    "Cyber Slavery": {
        "keywords": ["cyber slavery", "forced labor"],
        "subcategories": {}
    },

    # Crime Against Women & Children
    "Rape/Gang Rape": {
        "keywords": ["rape", "gang rape"],
        "subcategories": {}
    },
    "Sexual Harassment": {
        "keywords": ["sexual harassment", "harass"],
        "subcategories": {}
    },
    "Cyber Stalking": {
        "keywords": ["cyber stalk", "stalker"],
        "subcategories": {}
    },
    "Cyber Bullying": {
        "keywords": ["cyber bully", "bully"],
        "subcategories": {}
    },
    "Child Pornography/Child Sexual Abuse Material (CSAM)": {
        "keywords": ["child porn", "csam"],
        "subcategories": {}
    },
    "Publishing/transmitting obscene material/sexually explicit material": {
        "keywords": ["obscene material", "sexually explicit"],
        "subcategories": {}
    },
    "Computer-Generated CSAM": {
        "keywords": ["computer-generated csam", "synthetic child porn"],
        "subcategories": {}
    },
    "Fake Social Media Profile": {
        "keywords": ["fake social media", "fraudulent profile"],
        "subcategories": {}
    },
    "Cyber Blackmailing & Threatening": {
        "keywords": ["blackmail", "threaten"],
        "subcategories": {}
    },
    "Online Human Trafficking": {
        "keywords": ["human trafficking", "cyber trafficking"],
        "subcategories": {}
    },

    # Financial Crimes
    "Investment Scam/Trading Scam": {
        "keywords": ["investment scam", "trading fraud"],
        "subcategories": {}
    },
    "Online Job Fraud": {
        "keywords": ["job fraud", "employment scam"],
        "subcategories": {}
    },
    "Tech Support/Customer Care Scam": {
        "keywords": ["tech support scam", "customer care fraud"],
        "subcategories": {}
    },
    "Matrimonial/Romance Scam/Honey Trap": {
        "keywords": ["romance scam", "honey trap"],
        "subcategories": {}
    },
    "Impersonation of Government Authority/Other than Government Authority": {
        "keywords": ["impersonate", "fake authority"],
        "subcategories": {}
    },
    "SIM Swap Fraud": {
        "keywords": ["sim swap", "sim fraud"],
        "subcategories": {}
    },
    "Sextortion through Video Call": {
        "keywords": ["sextortion", "video call blackmail"],
        "subcategories": {}
    },
    "Aadhaar-enabled Payment System (AePS) Fraud": {
        "keywords": ["aeps fraud", "aadhaar fraud"],
        "subcategories": {}
    },
    "Identity Theft/Cloning": {
        "keywords": ["identity theft", "cloning"],
        "subcategories": {}
    },
    "Courier/Parcel Scam": {
        "keywords": ["courier scam", "parcel fraud"],
        "subcategories": {}
    },
    "Phishing": {
        "keywords": ["phish", "email fraud", "spoof"],
        "subcategories": {}
    },
    "Online Shopping/E-commerce Frauds": {
        "keywords": ["online shopping fraud", "ecommerce scam"],
        "subcategories": {}
    },
    "Advance Fee Fraud": {
        "keywords": ["advance fee", "upfront payment"],
        "subcategories": {}
    },
    "Real Estate/Rental Payment Fraud": {
        "keywords": ["real estate fraud", "rental scam"],
        "subcategories": {}
    }
}

# Step 3: Keyword-Based Categorization
def keyword_based_categorization(description):
    for category, details in categories.items():
        # Check if any keyword matches the description
        if any(keyword in description.lower() for keyword in details["keywords"]):
            return category
    return None

data["Category"] = data["description"].apply(keyword_based_categorization)

# Step 4: KNN-Based Categorization
# Prepare labeled data for training
labeled_data = [
    ("malware detected", "Malware Attack"),
    ("ransomware encrypted files", "Ransomware Attack"),
    ("website hacked", "Hacking/Defacement"),
    ("data breach occurred", "Data Breach/Theft"),
    ("fake profile created", "Fake profile"),
    ("phishing email received", "Phishing"),
    ("cyber terrorism threat", "Cyber Terrorism"),
    ("social media account hacked", "Social Media Account Hacking"),
    ("online gambling fraud", "Online Gambling/Betting Fraud"),
    ("business email compromised", "Business Email Compromise/Email Takeover"),
    ("provocative speech online", "Provocative Speech for unlawful acts"),
    ("matrimonial scam reported", "Matrimonial/Honey Trapping Scam"),
    ("fake news spread", "Fake News"),
    ("cyber stalking incident", "Cyber Stalking/Bullying"),
    ("defamation case filed", "Defamation"),
    ("cyber pornography found", "Cyber Pornography"),
    ("obscene material sent", "Sending obscene material"),
    ("ipr theft detected", "Intellectual Property (IPR) Thefts"),
    ("human trafficking online", "Cyber Enabled Human Trafficking"),
    ("cyber blackmailing reported", "Cyber Blackmailing & Threatening"),
    ("online piracy detected", "Online Piracy"),
    ("spoofing attack detected", "Spoofing"),
    ("cyber slavery reported", "Cyber Slavery"),
    ("rape case reported", "Rape/Gang Rape"),
    ("sexual harassment complaint", "Sexual Harassment"),
    ("cyber stalking incident", "Cyber Stalking"),
    ("cyber bullying incident", "Cyber Bullying"),
    ("child pornography found", "Child Pornography/Child Sexual Abuse Material (CSAM)"),
    ("obscene material published", "Publishing/transmitting obscene material/sexually explicit material"),
    ("computer-generated csam detected", "Computer-Generated CSAM"),
    ("fake social media profile", "Fake Social Media Profile"),
    ("cyber blackmailing reported", "Cyber Blackmailing & Threatening"),
    ("human trafficking online", "Online Human Trafficking"),
    ("investment scam detected", "Investment Scam/Trading Scam"),
    ("online job fraud reported", "Online Job Fraud"),
    ("tech support scam detected", "Tech Support/Customer Care Scam"),
    ("romance scam reported", "Matrimonial/Romance Scam/Honey Trap"),
    ("impersonation of authority", "Impersonation of Government Authority/Other than Government Authority"),
    ("sim swap fraud detected", "SIM Swap Fraud"),
    ("sextortion through video call", "Sextortion through Video Call"),
    ("aeps fraud detected", "Aadhaar-enabled Payment System (AePS) Fraud"),
    ("identity theft reported", "Identity Theft/Cloning"),
    ("courier scam detected", "Courier/Parcel Scam"),
    ("phishing email received", "Phishing"),
    ("online shopping fraud reported", "Online Shopping/E-commerce Frauds"),
    ("advance fee fraud detected", "Advance Fee Fraud"),
    ("real estate fraud detected", "Real Estate/Rental Payment Fraud")
]

# Split labeled data into features and labels
train_texts = [item[0] for item in labeled_data]
train_categories = [item[1] for item in labeled_data]

# Build KNN model for category prediction
category_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=3))

# Check if cross-validation is feasible
unique_categories = set(train_categories)
if len(train_texts) < 6 or any(train_categories.count(cat) < 2 for cat in unique_categories):
    logging.warning("Insufficient data for cross-validation. Skipping cross-validation.")
else:
    # Perform cross-validation
    try:
        cv_scores = cross_val_score(category_model, train_texts, train_categories, cv=3, scoring="accuracy")
        logging.info(f"Cross-validation accuracy: {np.mean(cv_scores):.2f}")
    except Exception as e:
        logging.warning(f"Cross-validation failed: {e}")

# Train the model
category_model.fit(train_texts, train_categories)

# Predict categories for uncategorized descriptions
uncategorized_mask = data["Category"].isnull()
if uncategorized_mask.any():
    uncategorized_texts = data.loc[uncategorized_mask, "description"]
    predicted_categories = category_model.predict(uncategorized_texts)

    # Update the DataFrame with predictions
    data.loc[uncategorized_mask, "Category"] = predicted_categories

# Log uncategorized descriptions
uncategorized_descriptions = data[data["Category"].isnull()]
if not uncategorized_descriptions.empty:
    logging.warning(f"{len(uncategorized_descriptions)} descriptions could not be categorized.")

# Step 5: Save the Output CSV
data.to_csv(output_file, index=False)

logging.info(f"Categorized data saved to {output_file}")