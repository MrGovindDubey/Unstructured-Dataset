import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Load the CSV file and keyword file
input_file = "cleaned_data.csv"
output_file = "categorized_cybercrime4.csv"
keyword_file = "keywords.json"

try:
    data = pd.read_csv(input_file)
    if "description" not in data.columns:
        raise ValueError("Input CSV must contain a 'description' column.")
except Exception as e:
    logging.error(f"Error loading input file: {e}")
    exit(1)

try:
    with open(keyword_file, "r") as f:
        categories = json.load(f)
except Exception as e:
    logging.error(f"Error loading keyword file: {e}")
    exit(1)

# Step 2: Advanced Keyword-Based Categorization
def keyword_based_categorization(description):
    description = description.lower()
    best_match_category = None
    best_match_subcategory = None
    best_match_score = 0

    for category, details in categories.items():
        # Check category keywords
        category_keywords = details.get("keywords", [])
        category_score = sum(len(keyword) for keyword in category_keywords if keyword in description)

        # Check subcategory keywords
        for subcategory, sub_keywords in details.get("subcategories", {}).items():
            subcategory_score = sum(len(keyword) for keyword in sub_keywords if keyword in description)

            # Prioritize subcategories over categories
            if subcategory_score > best_match_score:
                best_match_score = subcategory_score
                best_match_category = category
                best_match_subcategory = subcategory

        # Update category match if no subcategory match found
        if category_score > best_match_score and best_match_subcategory is None:
            best_match_score = category_score
            best_match_category = category
            best_match_subcategory = "Others"

    return best_match_category or "Uncategorized", best_match_subcategory or "Others"

data["Category"], data["Subcategory"] = zip(*data["description"].apply(keyword_based_categorization))

# Step 3: Post-Processing Rules
def post_processing_rules(row):
    description = row["description"].lower()
    category = row["Category"]
    subcategory = row["Subcategory"]

    # Rule 1: APK file scams should be categorized under Phishing
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

# Step 4: KNN-Based Categorization and Subcategorization
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
category_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=5, weights="distance"))

# Train the category model
category_model.fit(train_texts, train_categories)

# Filter out rows with None subcategories for subcategory model training
filtered_labeled_data = [
    (text, cat, subcat) for text, cat, subcat in labeled_data if subcat != "Others"
]
if filtered_labeled_data:
    train_texts_sub = [item[0] for item in filtered_labeled_data]
    train_subcategories_filtered = [item[2] for item in filtered_labeled_data]

    subcategory_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), KNeighborsClassifier(n_neighbors=5, weights="distance"))
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

# Step 5: Save the Output CSV
data.to_csv(output_file, index=False)

logging.info(f"Categorized data saved to {output_file}")