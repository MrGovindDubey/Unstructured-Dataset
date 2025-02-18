from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import torch
import time

# Set the GPU for processing (ensure device is available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# Ensure the column name is 'description'
descriptions = df["description"]

# Define categories and subcategories
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
candidate_labels = [
    subcategory for subcategories in categories.values() for subcategory in subcategories
]

# Load the paraphrase-MiniLM-L3-v2 model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=device)

# Encode the candidate labels once
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)

# Batch size for processing
batch_size = 1000
total_descriptions = len(descriptions)
results = []

print(f"Starting classification for {total_descriptions} descriptions...\n")

start_time = time.time()

# Process descriptions in batches
for start_idx in range(0, total_descriptions, batch_size):
    batch = descriptions[start_idx : start_idx + batch_size]
    
    # Encode batch descriptions
    description_embeddings = model.encode(batch.tolist(), convert_to_tensor=True)
    
    # Compute cosine similarity
    similarities = util.cos_sim(description_embeddings, label_embeddings)
    
    # Get the best matching label for each description
    best_indices = torch.argmax(similarities, dim=1)
    predicted_labels = [candidate_labels[idx] for idx in best_indices]

    # Find main categories for subcategories
    batch_results = []
    for desc, pred_label in zip(batch, predicted_labels):
        main_category = next(
            (cat for cat, subs in categories.items() if pred_label in subs), None
        )
        batch_results.append(
            {
                "description": desc,
                "main_category": main_category,
                "subcategory": pred_label,
            }
        )

    # Append batch results to the main list
    results.extend(batch_results)

    # Save intermediate results to avoid data loss
    with open(f"classified_batch_{start_idx}.json", "w") as f:
        json.dump(batch_results, f, indent=4)

    elapsed_time = time.time() - start_time
    print(
        f"Processed {start_idx + len(batch)}/{total_descriptions} descriptions... "
        f"Elapsed time: {elapsed_time:.2f} seconds"
    )

# Save the final result
with open("classified_descriptions.json", "w") as f:
    json.dump(results, f, indent=4)

end_time = time.time()
total_time = end_time - start_time

print("\nClassification completed!")
print(f"Total time taken: {total_time / 60:.2f} minutes")
print("Classified descriptions saved to 'classified_descriptions.json'")
