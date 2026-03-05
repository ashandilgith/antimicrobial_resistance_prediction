import pandas as pd
import os
import kagglehub
import glob  # <-- Added the glob import

print("Locating AMR dataset in cache...")
path = kagglehub.dataset_download("vihaankulkarni/antimicrobial-resistance-dataset")

# <-- CHANGED THIS SECTION: Automatically finds the CSV file -->
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Could not find a CSV file in the downloaded folder.")
csv_path = csv_files[0]
# <----------------------------------------------------------->

df = pd.read_csv(csv_path)

# Extract only the exact features the model was trained on
feature_cols = [col for col in df.columns if col.startswith('gene_')]

# Sample 5 actual rows to create a test file
mock_patients = df[feature_cols].sample(5, random_state=99)
mock_patients.to_csv("mock_patients.csv", index=False)

print("Success! Saved 5 real patient profiles to mock_patients.csv.")