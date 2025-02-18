from transformers import pipeline
from keybert import KeyBERT
import pandas as pd
import json
import time  # To measure time taken for each step

# Load the CSV file with descriptions
df = pd.read_csv('cleaned_data.csv')

# Initialize zero-shot classification pipeline with GPU
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Initialize KeyBERT model for keyword extraction
kw_model = KeyBERT()

# Ensure the column name is 'description'
descriptions = df['description']

# Define the four main categories and their subcategories
categories = {
    "Financial Crimes": [
        "Investment Scam/Trading Scam",
        "Online Job Fraud",
        # Add all subcategories
    ],
    "Crime Against Women & Children": [
        "Rape/Gang Rape",
        "Sexual Harassment",
        # Add all subcategories
    ],
    "Cyber Attack/Dependent Crimes": [
        "Malware Attack",
        "Ransomware Attack",
        # Add all subcategories
    ],
    "Other Cyber Crimes": [
        "Fake Profile",
        "Phishing",
        # Add all subcategories
    ]
}

# Flatten the categories for zero-shot classification
candidate_labels = [subcategory for subcategories in categories.values() for subcategory in subcategories]

# Function to classify and extract keywords
def classify_and_extract_keywords(text):
    # Classify the text into one of the candidate labels (subcategories)
    result = classifier(text, candidate_labels)
    predicted_subcategory = result['labels'][0]
    
    # Extract keywords using KeyBERT
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    extracted_keywords = [keyword for keyword, score in keywords]
    
    return predicted_subcategory, extracted_keywords

# Process descriptions in batches
batch_size = 500  # Adjust based on available memory
total_descriptions = len(descriptions)
all_results = []

print(f"Starting classification and keyword extraction for {total_descriptions} descriptions...\n")

start_time = time.time()  # Start timing

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
    
    # Save intermediate results
    with open(f'classified_keywords_checkpoint_{start_idx}.json', 'w') as f:
        json.dump(batch_results, f, indent=4)
    
    elapsed_time = time.time() - start_time
    print(f"Processed {start_idx + len(batch)}/{total_descriptions} descriptions... Elapsed time: {elapsed_time:.2f} seconds")

# Save the final result
with open('classified_keywords.json', 'w') as f:
    json.dump(all_results, f, indent=4)

end_time = time.time()
total_time = end_time - start_time

print("\nClassification and keyword extraction completed!")
print(f"Total time taken: {total_time:.2f} seconds")
print("Classified descriptions and extracted keywords saved to 'classified_keywords.json'")
