import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download("vihaankulkarni/antimicrobial-resistance-dataset")
csv_path = os.path.join(path, "Kaggle_AMR_Dataset_v1.0_final.csv")
df = pd.read_csv(csv_path)

# Extract only the exact features the model expects
feature_cols = [col for col in df.columns if col.startswith('gene_')]

# Slice 5 real rows to create a test file
mock_patients = df[feature_cols].sample(5, random_state=99)
mock_patients.to_csv("mock_patients.csv", index=False)
print("Saved 5 real patient profiles to mock_patients.csv.")