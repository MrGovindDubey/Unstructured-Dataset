import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
input_file = r'D:\Projects\Python\structured_output.csv'  # Update this with your actual file path
df = pd.read_csv(input_file)

# Generate a detailed EDA report
profile = ProfileReport(df, title="Exploratory Data Analysis Report", explorative=True)

# Save the report to an HTML file
output_file = r'D:\Projects\Python\eda_report.html'  # Update the path for the output report
profile.to_file(output_file)

print(f"Report generated and saved to {output_file}")
