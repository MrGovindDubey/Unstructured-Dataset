import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
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

# Step 2: Define predefined categories and subcategories with refined keywords
categories = {
    # Crime Against Women & Children
    "Crime Against Women & Children": {
        "keywords": ["women", "child", "rape", "harassment", "stalking", "bullying", "csam", "obscene", "blackmail", "trafficking"],
        "subcategories": {
            "Rape/Gang Rape": ["rape", "gang rape"],
            "Sexual Harassment": ["sexual harassment", "harass"],
            "Cyber Stalking": ["cyber stalk", "stalker"],
            "Cyber Bullying": ["cyber bully", "bully"],
            "Child Pornography/Child Sexual Abuse Material (CSAM)": ["child porn", "csam"],
            "Publishing/transmitting obscene material/sexually explicit material": ["obscene material", "sexually explicit"],
            "Computer-Generated CSAM": ["computer-generated csam", "synthetic child porn"],
            "Fake Social Media Profile": ["fake social media", "fraudulent profile"],
            "Cyber Blackmailing & Threatening": ["blackmail", "threaten"],
            "Online Human Trafficking": ["human trafficking", "cyber trafficking"]
        }
    },

    # Financial Crimes
    "Financial Crimes": {
        "keywords": ["investment scam", "job fraud", "tech support", "romance scam", "impersonation", "sim swap", "sextortion", "aeps fraud", "identity theft", "courier scam", "phishing", "shopping fraud", "advance fee", "real estate fraud", "apk file scam", "net banking fraud"],
        "subcategories": {
            "Investment Scam/Trading Scam": ["investment scam", "trading fraud"],
            "Online Job Fraud": ["job fraud", "employment scam"],
            "Tech Support/Customer Care Scam": ["tech support scam", "customer care fraud"],
            "Matrimonial/Romance Scam/Honey Trap": ["romance scam", "honey trap"],
            "Impersonation of Government Authority/Other than Government Authority": ["impersonate", "fake authority"],
            "SIM Swap Fraud": ["sim swap", "sim fraud"],
            "Sextortion through Video Call": ["sextortion", "video call blackmail"],
            "Aadhaar-enabled Payment System (AePS) Fraud": ["aeps fraud", "aadhaar fraud", "aadhaar payment"],
            "Identity Theft/Cloning": ["identity theft", "cloning"],
            "Courier/Parcel Scam": ["courier scam", "parcel fraud", "delivery scam"],
            "Phishing": ["phish", "email fraud", "spoof"],
            "Online Shopping/E-commerce Frauds": ["online shopping fraud", "ecommerce scam", "shopping fraud"],
            "Advance Fee Fraud": ["advance fee", "upfront payment"],
            "Real Estate/Rental Payment Fraud": ["real estate fraud", "rental scam", "property fraud"]
        }
    },

    # Cyber Attack/Dependent Crimes
    "Cyber Attack/Dependent Crimes": {
        "keywords": ["malware", "ransomware", "hack", "data breach", "tamper", "dos", "ddos"],
        "subcategories": {
            "Malware Attack": ["malware", "virus", "trojan"],
            "Ransomware Attack": ["ransomware", "encrypt files"],
            "Hacking/Defacement": ["hack", "deface", "breach"],
            "Data Breach/Theft": ["data breach", "data theft", "leak"],
            "Tampering with computer source documents": ["tamper", "source code", "modify"],
            "Denial of Service (DoS)/Distributed Denial of Service (DDoS) attacks": ["dos", "ddos", "service attack"]
        }
    },

    # Other Cyber Crimes
    "Other Cyber Crimes": {
        "keywords": ["fake profile", "phishing", "cyber terror", "social media hack", "gambling", "email compromise", "provocative speech", "matrimonial scam", "fake news", "cyber stalking", "defamation", "cyber pornography", "obscene", "ipr theft", "human trafficking", "blackmail", "piracy", "spoof", "cyber slavery", "apk file scam"],
        "subcategories": {
            "Fake profile": ["fake profile", "fraudulent account"],
            "Phishing": ["phish", "email fraud", "spoof"],
            "Cyber Terrorism": ["cyber terror", "terrorist"],
            "Social Media Account Hacking": ["social media hack", "account takeover"],
            "Online Gambling/Betting Fraud": ["gambling", "betting fraud"],
            "Business Email Compromise/Email Takeover": ["email compromise", "email takeover"],
            "Provocative Speech for unlawful acts": ["provocative speech", "unlawful act"],
            "Matrimonial/Honey Trapping Scam": ["matrimonial scam", "honey trap"],
            "Fake News": ["fake news", "false information"],
            "Cyber Stalking/Bullying": ["cyber stalk", "cyber bully"],
            "Defamation": ["defame", "character assassination"],
            "Cyber Pornography": ["cyber porn", "online pornography"],
            "Sending obscene material": ["obscene", "sexually explicit"],
            "Intellectual Property (IPR) Thefts": ["ipr theft", "copyright violation"],
            "Cyber Enabled Human Trafficking": ["human trafficking", "cyber trafficking"],
            "Cyber Blackmailing & Threatening": ["blackmail", "threaten"],
            "Online Piracy": ["piracy", "illegal download"],
            "Spoofing": ["spoof", "fake identity"],
            "Cyber Slavery": ["cyber slavery", "forced labor"]
        }
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
            return category, "Others"  # Return category if no subcategory matches
    return "Uncategorized", "Others"  # Return Uncategorized if no match found

data["Category"], data["Subcategory"] = zip(*data["description"].apply(keyword_based_categorization))

# Step 4: Post-Processing Rules
def post_processing_rules(row):
    description = row["description"].lower()
    category = row["Category"]
    subcategory = row["Subcategory"]

    # Rule 1: APK file scams should not be categorized under Courier/Parcel Scam
    if "apk file" in description and category == "Financial Crimes" and subcategory == "Courier/Parcel Scam":
        return "Other Cyber Crimes", "Phishing"

    # Rule 2: Net banking frauds involving Aadhaar should be categorized under AePS Fraud
    if "net banking" in description and "aadhaar" in description:
        return "Financial Crimes", "Aadhaar-enabled Payment System (AePS) Fraud"

    # Rule 3: Fake APK file scams should be categorized under Phishing
    if "apk file" in description and category == "Other Cyber Crimes" and subcategory == "Fake News":
        return "Other Cyber Crimes", "Phishing"

    return category, subcategory

data[["Category", "Subcategory"]] = data.apply(post_processing_rules, axis=1, result_type="expand")

# Step 5: KNN-Based Categorization and Subcategorization
# Prepare labeled data for training
labeled_data = [
    ("malware detected", "Cyber Attack/Dependent Crimes", "Malware Attack"),
    ("ransomware encrypted files", "Cyber Attack/Dependent Crimes", "Ransomware Attack"),
    ("website hacked", "Cyber Attack/Dependent Crimes", "Hacking/Defacement"),
    ("data breach occurred", "Cyber Attack/Dependent Crimes", "Data Breach/Theft"),
    ("fake profile created", "Other Cyber Crimes", "Fake profile"),
    ("phishing email received", "Other Cyber Crimes", "Phishing"),
    ("cyber terrorism threat", "Other Cyber Crimes", "Cyber Terrorism"),
    ("social media account hacked", "Other Cyber Crimes", "Social Media Account Hacking"),
    ("online gambling fraud", "Other Cyber Crimes", "Online Gambling/Betting Fraud"),
    ("business email compromised", "Other Cyber Crimes", "Business Email Compromise/Email Takeover"),
    ("provocative speech online", "Other Cyber Crimes", "Provocative Speech for unlawful acts"),
    ("matrimonial scam reported", "Other Cyber Crimes", "Matrimonial/Honey Trapping Scam"),
    ("fake news spread", "Other Cyber Crimes", "Fake News"),
    ("cyber stalking incident", "Other Cyber Crimes", "Cyber Stalking/Bullying"),
    ("defamation case filed", "Other Cyber Crimes", "Defamation"),
    ("cyber pornography found", "Other Cyber Crimes", "Cyber Pornography"),
    ("obscene material sent", "Other Cyber Crimes", "Sending obscene material"),
    ("ipr theft detected", "Other Cyber Crimes", "Intellectual Property (IPR) Thefts"),
    ("human trafficking online", "Other Cyber Crimes", "Cyber Enabled Human Trafficking"),
    ("cyber blackmailing reported", "Other Cyber Crimes", "Cyber Blackmailing & Threatening"),
    ("online piracy detected", "Other Cyber Crimes", "Online Piracy"),
    ("spoofing attack detected", "Other Cyber Crimes", "Spoofing"),
    ("cyber slavery reported", "Other Cyber Crimes", "Cyber Slavery"),
    ("rape case reported", "Crime Against Women & Children", "Rape/Gang Rape"),
    ("sexual harassment complaint", "Crime Against Women & Children", "Sexual Harassment"),
    ("cyber stalking incident", "Crime Against Women & Children", "Cyber Stalking"),
    ("cyber bullying incident", "Crime Against Women & Children", "Cyber Bullying"),
    ("child pornography found", "Crime Against Women & Children", "Child Pornography/Child Sexual Abuse Material (CSAM)"),
    ("obscene material published", "Crime Against Women & Children", "Publishing/transmitting obscene material/sexually explicit material"),
    ("computer-generated csam detected", "Crime Against Women & Children", "Computer-Generated CSAM"),
    ("fake social media profile", "Crime Against Women & Children", "Fake Social Media Profile"),
    ("cyber blackmailing reported", "Crime Against Women & Children", "Cyber Blackmailing & Threatening"),
    ("human trafficking online", "Crime Against Women & Children", "Online Human Trafficking"),
    ("investment scam detected", "Financial Crimes", "Investment Scam/Trading Scam"),
    ("online job fraud reported", "Financial Crimes", "Online Job Fraud"),
    ("tech support scam detected", "Financial Crimes", "Tech Support/Customer Care Scam"),
    ("romance scam reported", "Financial Crimes", "Matrimonial/Romance Scam/Honey Trap"),
    ("impersonation of authority", "Financial Crimes", "Impersonation of Government Authority/Other than Government Authority"),
    ("sim swap fraud detected", "Financial Crimes", "SIM Swap Fraud"),
    ("sextortion through video call", "Financial Crimes", "Sextortion through Video Call"),
    ("aeps fraud detected", "Financial Crimes", "Aadhaar-enabled Payment System (AePS) Fraud"),
    ("identity theft reported", "Financial Crimes", "Identity Theft/Cloning"),
    ("courier scam detected", "Financial Crimes", "Courier/Parcel Scam"),
    ("phishing email received", "Financial Crimes", "Phishing"),
    ("online shopping fraud reported", "Financial Crimes", "Online Shopping/E-commerce Frauds"),
    ("advance fee fraud detected", "Financial Crimes", "Advance Fee Fraud"),
    ("real estate fraud detected", "Financial Crimes", "Real Estate/Rental Payment Fraud")
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
    (text, cat, subcat) for text, cat, subcat in labeled_data if subcat != "Others"
]
if filtered_labeled_data:
    train_texts_sub = [item[0] for item in filtered_labeled_data]
    train_subcategories_filtered = [item[2] for item in filtered_labeled_data]

    subcategory_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=3))
    subcategory_model.fit(train_texts_sub, train_subcategories_filtered)
else:
    logging.warning("No subcategories available for training the subcategory model.")

# Predict categories and subcategories for uncategorized descriptions
uncategorized_mask = data["Category"] == "Uncategorized"
if uncategorized_mask.any():
    uncategorized_texts = data.loc[uncategorized_mask, "description"]
    predicted_categories = category_model.predict(uncategorized_texts)

    # Update the DataFrame with predictions
    data.loc[uncategorized_mask, "Category"] = predicted_categories

# Predict subcategories for descriptions with categories but no subcategories
subcategory_mask = (data["Category"] != "Uncategorized") & (data["Subcategory"] == "Others")
if subcategory_mask.any() and filtered_labeled_data:
    uncategorized_texts_sub = data.loc[subcategory_mask, "description"]
    predicted_subcategories = subcategory_model.predict(uncategorized_texts_sub)

    # Update the DataFrame with predictions
    data.loc[subcategory_mask, "Subcategory"] = predicted_subcategories

# Log uncategorized descriptions
uncategorized_descriptions = data[data["Category"] == "Uncategorized"]
if not uncategorized_descriptions.empty:
    logging.warning(f"{len(uncategorized_descriptions)} descriptions could not be categorized.")

# Step 6: Save the Output CSV
data.to_csv(output_file, index=False)

logging.info(f"Categorized data saved to {output_file}")