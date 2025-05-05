import pandas as pd

# Step 1: Read the Excel file, skipping the top 5 rows
input_path = "data/test.xls"
df = pd.read_excel(input_path, skiprows=5)

# Step 2: Save the cleaned data to a CSV file
output_path = "data/test_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"[INFO] Cleaned data saved to {output_path}")
