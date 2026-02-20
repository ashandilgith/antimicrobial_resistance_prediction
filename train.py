import pandas as pd
import kagglehub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Ingest Actual Data
print("Downloading real-world AMR dataset...")
path = kagglehub.dataset_download("vihaankulkarni/antimicrobial-resistance-dataset")
csv_path = os.path.join(path, "Kaggle_AMR_Dataset_v1.0_final.csv")
df = pd.read_csv(csv_path)

# 2. Feature Engineering
# X represents our inputs (the genetic markers)
feature_cols = [col for col in df.columns if col.startswith('gene_')]
X = df[feature_cols]

# y represents our target (did the bacteria survive the antibiotic?)
y = df['class_carbapenem']

# Split 80% of data to learn, 20% to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Tracking
mlflow.set_experiment("AMR_Carbapenem_Predictor")

with mlflow.start_run():
    # Train a Random Forest with 100 decision trees
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate Accuracy
    acc = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    
    # Save the physical model artifact
    mlflow.sklearn.log_model(clf, "random_forest_amr_model")
    print(f"Training Complete - Accuracy: {acc:.2f}")