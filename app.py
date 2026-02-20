import streamlit as st
import pandas as pd
import mlflow.sklearn
import os

st.set_page_config(page_title="AMR Predictor", page_icon="🧬")
st.title("Carbapenem Resistance Predictor")

# 1. Load the Model into Memory
@st.cache_resource
def load_model():
    model_path = "random_forest_amr_model"
    if os.path.exists(model_path):
        return mlflow.sklearn.load_model(model_path)
    return None

model = load_model()

if not model:
    st.error("Run train.py first to generate the model.")
    st.stop()

# 2. File Upload Interface
uploaded_file = st.file_uploader("Upload genomic features (CSV)", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    if st.button("Predict Resistance"):
        feature_cols = [col for col in input_data.columns if col.startswith('gene_')]
        
        # 3. Inference
        try:
            predictions = model.predict(input_data[feature_cols])
            for i, pred in enumerate(predictions):
                if pred == 1:
                    st.error(f"Sample {i+1}: HIGH RISK - Resistant to Carbapenem")
                else:
                    st.success(f"Sample {i+1}: Susceptible")
        except Exception as e:
            st.error(f"Prediction error. Details: {e}")