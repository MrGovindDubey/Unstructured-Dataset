from keybert import KeyBERT
import pandas as pd
import json

# Load the CSV file with descriptions
df = pd.read_csv('cleaned_data.csv')

# Initialize KeyBERT model
kw_model = KeyBERT()

# Ensure the column name is 'description'
descriptions = df['description']

# Define category-specific keywords/phrases with subcategories
category_keywords = {
    "Financial Crimes": {
        "Investment Scam/Trading Scam": ["investment scam", "trading scam"],
        "Online Job Fraud": ["online job fraud"],
        "Tech Support/Customer Care Scam": ["tech support scam", "customer care scam"],
        "Online Loan Fraud": ["online loan fraud"],
        "Matrimonial/Romance Scam/Honey Trapping Scam": ["matrimonial scam", "romance scam", "honey trapping scam"],
        "Impersonation of Govt. Servant": ["impersonation of govt servant"],
        "Cheating by Impersonation (Other than Government Servant)": ["cheating by impersonation"],
        "SIM Swap Fraud": ["sim swap fraud"],
        "Sextortion/Nude Video Call": ["sextortion", "nude video call"],
        "Aadhaar-enabled Payment System (AePS) Fraud / Biometric Cloning": ["aeps fraud", "biometric cloning"],
        "Identity Theft": ["identity theft"],
        "Courier/Parcel Scam": ["courier scam", "parcel scam"],
        "Phishing": ["phishing"],
        "Online Shopping/E-commerce Frauds": ["online shopping fraud", "ecommerce fraud"],
        "Advance Fee Fraud": ["advance fee fraud"],
        "Real Estate/Rental Payment Fraud": ["real estate fraud", "rental payment fraud"],
        "Others (in Financial Crimes)": []
    },
    "Crime Against Women & Children": {
        "Rape/Gang Rape": ["rape", "gang rape"],
        "Sexual Harassment": ["sexual harassment"],
        "Cyber Stalking": ["cyber stalking"],
        "Cyber Bullying": ["cyber bullying"],
        "Child Pornography/Child Sexual Abuse Material (CSAM)": ["child pornography", "csam"],
        "Publishing/Transmitting Obscene Material/Sexually Explicit Material": ["obscene material", "sexually explicit material"],
        "Computer-Generated CSAM/CSEM": ["computer-generated csam", "csem"],
        "Fake Social Media Profile": ["fake social media profile"],
        "Cyber Blackmailing & Threatening": ["cyber blackmailing", "cyber threatening"],
        "Online Human Trafficking": ["online human trafficking"],
        "Cyber Voyeurism": ["cyber voyeurism"],
        "Defamation": ["defamation"],
        "Others (in Crime Against Women & Children)": []
    },
    # Add other categories and subcategories similarly...
}

# Function to extract and filter keywords based on categories and subcategories
def extract_and_filter_keywords(text):
    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    
    # Flatten the list of keywords
    extracted_keywords = [keyword for keyword, score in keywords]
    
    # Filter keywords based on category-specific keywords
    filtered_keywords = {}
    for category, subcategories in category_keywords.items():
        filtered_keywords[category] = {}
        for subcategory, subcat_keywords in subcategories.items():
            filtered_keywords[category][subcategory] = []
            for keyword in extracted_keywords:
                if any(subcat_keyword.lower() in keyword.lower() for subcat_keyword in subcat_keywords):
                    filtered_keywords[category][subcategory].append(keyword)
    
    return filtered_keywords

# Apply the function to each description and collect all results
all_filtered_keywords = []
for description in descriptions:
    filtered_keywords = extract_and_filter_keywords(description)
    all_filtered_keywords.append(filtered_keywords)

# Save the result to a JSON file
with open('filtered_keywords.json', 'w') as f:
    json.dump(all_filtered_keywords, f, indent=4)

print("Filtered keywords extracted and saved to 'filtered_keywords.json'")