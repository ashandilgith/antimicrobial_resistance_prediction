import streamlit as st
import pandas as pd
import mlflow.sklearn
import os

# --- BRANDING CONFIGURATION ---
APP_NAME = "AMR-72x"
APP_TAGLINE = "Instant Genomic Carbapenem Resistance Prediction"
# ------------------------------

# 1. Page Configuration & Styling
st.set_page_config(page_title=APP_NAME, page_icon="🧬", layout="wide")

st.markdown(f"""
    <style>
    .main-header {{ font-size: 2.8rem; color: #0F172A; font-weight: 800; margin-bottom: 0; letter-spacing: -1px; }}
    .sub-header {{ font-size: 1.2rem; color: #475569; margin-bottom: 2.5rem; font-weight: 500; }}
    .brand-text {{ color: #2563EB; }}
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar - Education & Instructions
with st.sidebar:
    st.markdown(f"<h2 style='color: #1E3A8A;'>🧬 {APP_NAME} Engine</h2>", unsafe_allow_html=True)
    st.write(
        "**Carbapenem** is a 'last resort' antibiotic. "
        "If a patient has a severe infection, doctors need to know immediately "
        "if the bacteria has evolved to resist it."
    )
    
    st.info(
        f"**The {APP_NAME} Advantage:**\n"
        "Bypasses the standard 72-hour lab culture wait time by analyzing bacterial DNA directly. "
        "Upload a genomic sequence (CSV) to generate an instant resistance probability."
    )
    
    st.header(" Diagnostic Test")
    st.write("Drag and drop `mock_patients.csv` into the main interface to simulate a clinical run.")

# 3. Main Interface Header
st.markdown(f'<p class="main-header"><span class="brand-text">{APP_NAME}</span> | Clinical Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{APP_TAGLINE}</p>', unsafe_allow_html=True)

# Load Model (Cached for speed)
@st.cache_resource
def load_model():
    model_path = "random_forest_amr_model"
    if os.path.exists(model_path):
        return mlflow.sklearn.load_model(model_path)
    return None

model = load_model()

if not model:
    st.error(f"⚠️ {APP_NAME} engine offline. Model artifact not found. Please run train.py first.")
    st.stop()

# 4. Interactive File Upload Area
st.write("###  Ingest Genomic Sequence")
uploaded_file = st.file_uploader("Upload genomic features (CSV format)", type="csv", label_visibility="collapsed")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    
    with st.expander(" View Raw Patient Data Profiles"):
        st.dataframe(input_data.head(), use_container_width=True)
    
    st.write("---")
    st.write("### 🩺 Diagnostic Engine")
    
    if st.button(f"Run {APP_NAME} Analysis", type="primary", use_container_width=True):
        feature_cols = [col for col in input_data.columns if col.startswith('gene_')]
        
        if len(feature_cols) == 0:
            st.error("Invalid CSV: No genetic markers (columns starting with 'gene_') found.")
        else:
            try:
                with st.spinner(f'{APP_NAME} is analyzing genetic markers...'):
                    predictions = model.predict(input_data[feature_cols])
                
                st.write("#### Clinical Results:")
                
                cols = st.columns(3)
                for i, pred in enumerate(predictions):
                    col = cols[i % 3] 
                    with col:
                        if pred == 1:
                            st.error(f"**Sample {i+1}**\n\n🚨 HIGH RISK\n\nResistant to Carbapenem")
                        else:
                            st.success(f"**Sample {i+1}**\n\n✅ CLEAR\n\nSusceptible to Carbapenem")
                            
            except Exception as e:
                st.error(f"Prediction error. Details: {e}")
else:
    st.info("👆 Please upload a patient CSV file to begin analysis.")