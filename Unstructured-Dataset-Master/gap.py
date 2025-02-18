import pandas as pd

# Load CSV file (replace 'your_file.csv' with the actual filename)
df = pd.read_csv(r'C:\Users\Asus\Documents\un.csv')

# Keep only the 'description' column (Change if your column name is different)
df = df[['crime aditional information']]

# Drop rows where 'description' is empty or NaN
df.dropna(subset=['crime aditional information'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data to a new CSV file
df.to_csv("cleaned_data.csv", index=False)

print("Cleaned data saved as 'cleaned_data.csv'")
