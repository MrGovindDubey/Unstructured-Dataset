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
output_file = "categorized_cybercrime3.csv"

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
        "subcategories": {
            "Ransomware": ["ransomware", "encrypt files"],
            "Spyware": ["spyware", "track", "monitor"]
        }
    },
    "Ransomware Attack": {
        "keywords": ["ransomware", "encrypt files"],
        "subcategories": {}
    },
    "Hacking/Defacement": {
        "keywords": ["hack", "deface", "breach"],
        "subcategories": {
            "Brute Force Attack": ["brute force", "password guess"],
            "SQL Injection": ["sql injection", "database attack"]
        }
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
        "subcategories": {
            "Email Phishing": ["email", "mail"],
            "SMS Phishing": ["sms", "text message"]
        }
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
        "subcategories": {
            "Email Phishing": ["email", "mail"],
            "SMS Phishing": ["sms", "text message"]
        }
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

# Step 3: Keyword-Based Categorization and Subcategorization
def keyword_based_categorization(description):
    for category, details in categories.items():
        # Check if any keyword matches the description
        if any(keyword in description.lower() for keyword in details["keywords"]):
            # Check for subcategories
            for subcategory, sub_keywords in details["subcategories"].items():
                if any(sub_keyword in description.lower() for sub_keyword in sub_keywords):
                    return category, subcategory
            return category, None  # Return category if no subcategory matches
    return None, None  # Return None if no match found

data["Category"], data["Subcategory"] = zip(*data["description"].apply(keyword_based_categorization))

# Step 4: KNN-Based Categorization and Subcategorization
# Prepare labeled data for training
labeled_data = [
    ("malware detected", "Malware Attack", None),
    ("ransomware encrypted files", "Ransomware Attack", None),
    ("website hacked", "Hacking/Defacement", "Brute Force Attack"),
    ("data breach occurred", "Data Breach/Theft", None),
    ("fake profile created", "Fake profile", None),
    ("phishing email received", "Phishing", "Email Phishing"),
    ("cyber terrorism threat", "Cyber Terrorism", None),
    ("social media account hacked", "Social Media Account Hacking", None),
    ("online gambling fraud", "Online Gambling/Betting Fraud", None),
    ("business email compromised", "Business Email Compromise/Email Takeover", None),
    ("provocative speech online", "Provocative Speech for unlawful acts", None),
    ("matrimonial scam reported", "Matrimonial/Honey Trapping Scam", None),
    ("fake news spread", "Fake News", None),
    ("cyber stalking incident", "Cyber Stalking/Bullying", None),
    ("defamation case filed", "Defamation", None),
    ("cyber pornography found", "Cyber Pornography", None),
    ("obscene material sent", "Sending obscene material", None),
    ("ipr theft detected", "Intellectual Property (IPR) Thefts", None),
    ("human trafficking online", "Cyber Enabled Human Trafficking", None),
    ("cyber blackmailing reported", "Cyber Blackmailing & Threatening", None),
    ("online piracy detected", "Online Piracy", None),
    ("spoofing attack detected", "Spoofing", None),
    ("cyber slavery reported", "Cyber Slavery", None),
    ("rape case reported", "Rape/Gang Rape", None),
    ("sexual harassment complaint", "Sexual Harassment", None),
    ("cyber stalking incident", "Cyber Stalking", None),
    ("cyber bullying incident", "Cyber Bullying", None),
    ("child pornography found", "Child Pornography/Child Sexual Abuse Material (CSAM)", None),
    ("obscene material published", "Publishing/transmitting obscene material/sexually explicit material", None),
    ("computer-generated csam detected", "Computer-Generated CSAM", None),
    ("fake social media profile", "Fake Social Media Profile", None),
    ("cyber blackmailing reported", "Cyber Blackmailing & Threatening", None),
    ("human trafficking online", "Online Human Trafficking", None),
    ("investment scam detected", "Investment Scam/Trading Scam", None),
    ("online job fraud reported", "Online Job Fraud", None),
    ("tech support scam detected", "Tech Support/Customer Care Scam", None),
    ("romance scam reported", "Matrimonial/Romance Scam/Honey Trap", None),
    ("impersonation of authority", "Impersonation of Government Authority/Other than Government Authority", None),
    ("sim swap fraud detected", "SIM Swap Fraud", None),
    ("sextortion through video call", "Sextortion through Video Call", None),
    ("aeps fraud detected", "Aadhaar-enabled Payment System (AePS) Fraud", None),
    ("identity theft reported", "Identity Theft/Cloning", None),
    ("courier scam detected", "Courier/Parcel Scam", None),
    ("phishing email received", "Phishing", "Email Phishing"),
    ("online shopping fraud reported", "Online Shopping/E-commerce Frauds", None),
    ("advance fee fraud detected", "Advance Fee Fraud", None),
    ("real estate fraud detected", "Real Estate/Rental Payment Fraud", None)
]

# Split labeled data into features and labels
train_texts = [item[0] for item in labeled_data]
train_categories = [item[1] for item in labeled_data]
train_subcategories = [item[2] for item in labeled_data]

# Build KNN models for category and subcategory prediction
category_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=3))

# Train the category model
category_model.fit(train_texts, train_categories)

# Filter out rows with None subcategories for subcategory model training
filtered_labeled_data = [
    (text, cat, subcat) for text, cat, subcat in labeled_data if subcat is not None
]
if filtered_labeled_data:
    train_texts_sub = [item[0] for item in filtered_labeled_data]
    train_subcategories_filtered = [item[2] for item in filtered_labeled_data]

    subcategory_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=3))
    subcategory_model.fit(train_texts_sub, train_subcategories_filtered)
else:
    logging.warning("No subcategories available for training the subcategory model.")

# Predict categories and subcategories for uncategorized descriptions
uncategorized_mask = data["Category"].isnull()
if uncategorized_mask.any():
    uncategorized_texts = data.loc[uncategorized_mask, "description"]
    predicted_categories = category_model.predict(uncategorized_texts)

    # Update the DataFrame with predictions
    data.loc[uncategorized_mask, "Category"] = predicted_categories

# Predict subcategories for descriptions with categories but no subcategories
subcategory_mask = (data["Category"].notnull()) & (data["Subcategory"].isnull())
if subcategory_mask.any() and filtered_labeled_data:
    uncategorized_texts_sub = data.loc[subcategory_mask, "description"]
    predicted_subcategories = subcategory_model.predict(uncategorized_texts_sub)

    # Update the DataFrame with predictions
    data.loc[subcategory_mask, "Subcategory"] = predicted_subcategories

# Log uncategorized descriptions
uncategorized_descriptions = data[data["Category"].isnull()]
if not uncategorized_descriptions.empty:
    logging.warning(f"{len(uncategorized_descriptions)} descriptions could not be categorized.")

# Step 5: Save the Output CSV
data.to_csv(output_file, index=False)

logging.info(f"Categorized data saved to {output_file}")