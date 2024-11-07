import pandas as pd

# Load the dataset
input_file = r'D:\Projects\Python\structured_output.csv'  # Update the path to your CSV file
df = pd.read_csv(input_file)

# Basic information about the dataset
print("Basic Information:")
print(df.info())  # Data types and non-null counts
print("\n")

# Summary statistics for numerical columns
print("Summary Statistics for Numerical Columns:")
print(df.describe())  # Provides count, mean, std, min, 25%, 50%, 75%, max for numerical columns
print("\n")

# Count of unique values for categorical columns
print("Unique Value Counts for Categorical Columns:")
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())  # Count of unique values in each categorical column
print("\n")

# Checking for missing values
print("Missing Data Information:")
print(df.isnull().sum())  # Count of missing values in each column
print("\n")

# Additional: Checking for duplicates
print("Duplicate Rows in the Dataset:")
print(df.duplicated().sum())  # Count of duplicate rows
print("\n")

# Optionally, save the report to a text file
report_file = r'D:\Projects\Python\dataset_report.txt'  # Output file for the report
with open(report_file, 'w') as report:
    report.write("Basic Information:\n")
    report.write(str(df.info()) + "\n\n")
    report.write("Summary Statistics for Numerical Columns:\n")
    report.write(str(df.describe()) + "\n\n")
    report.write("Unique Value Counts for Categorical Columns:\n")
    for col in categorical_columns:
        report.write(f"\n{col}:\n")
        report.write(str(df[col].value_counts()) + "\n")
    report.write("\nMissing Data Information:\n")
    report.write(str(df.isnull().sum()) + "\n\n")
    report.write("Duplicate Rows in the Dataset:\n")
    report.write(str(df.duplicated().sum()) + "\n")

print(f"Report saved to {report_file}")
