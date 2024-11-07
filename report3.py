from ydata_profiling import ProfileReport
import pandas as pd

# Load your dataset (update with correct file path)
input_file = r'D:\Projects\Python\structured_output.csv'  # Update with your actual file path
df = pd.read_csv(input_file)

# Generate the Pandas Profiling report with valid parameters
profile = ProfileReport(df,
                        title="Train EDA Report",
                        explorative=True,  # Enables the most detailed analysis
                        minimal=True,  # Reduces the report size for quick overview
                        correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},  # Adds more correlation types
                        missing_diagrams={"heatmap": True, "dendrogram": True},  # Adds more visualizations for missing data
                        interactions={"continuous": True, "categorical": True})  # Includes interactions between continuous and categorical features

# Save the report to an HTML file
output_file = r'D:\Projects\Python\pandas_profiling_report.html'  # Update with your desired output file path
profile.to_file(output_file)

print(f"Pandas Profiling EDA report saved to {output_file}")
