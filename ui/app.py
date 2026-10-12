import streamlit as st
import requests
import base64
import time
from PIL import Image
import io
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/authenticate")

st.set_page_config(page_title="Deepfake Spectral Authenticator", layout="wide")

st.title("🛡️ Enterprise Deepfake Spectral Authenticator")
st.markdown("### Powered by Spatial Vision Transformers & Spectral Meta-Classification")

uploaded_file = st.file_uploader("Upload an image for forensic analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)
    
    if st.button("Run Forensic Analysis"):
        with st.spinner("Executing Meta-Classifier ensemble logic and XAI pipelines..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                start_time = time.time()
                response = requests.post(API_URL, files=files)
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Analysis complete in {end_time - start_time:.2f} seconds")
                    
                    verdict_col, conf_col = st.columns(2)
                    with verdict_col:
                        color = "red" if data["verdict"] == "Deepfake" else "green"
                        st.markdown(f"## Verdict: <span style='color:{color}'>{data['verdict']}</span>", unsafe_allow_html=True)
                    with conf_col:
                        st.markdown(f"## Confidence: {data['confidence'] * 100:.2f}%")
                    
                    st.markdown("---")
                    st.markdown("### Smart Ensemble Architecture")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Spatial Score (MobileViT)", f"{data['spatial_score'] * 100:.2f}%")
                    col2.metric("Spectral Score (XGBoost)", f"{data['spectral_score'] * 100:.2f}%")
                    col3.metric("Meta-Classifier (Final)", f"{data['ensemble_score'] * 100:.2f}%")
                    
                    st.markdown("---")
                    st.markdown("### Explainable AI (XAI) Evidence")
                    xai_tabs = st.tabs(["Spatial (Grad-CAM)", "Spectral (Tree SHAP)", "Frequency Heatmaps"])
                    
                    with xai_tabs[0]:
                        st.markdown("#### Spatial Manipulation Thermal Map (MobileViT)")
                        if data["xai"].get("spatial_gradcam"):
                            st.image(Image.open(io.BytesIO(base64.b64decode(data["xai"]["spatial_gradcam"]))), use_column_width=True)
                        else:
                            st.warning("Grad-CAM visualization not available.")
                            
                    with xai_tabs[1]:
                        st.markdown("#### Feature Importance (XGBoost)")
                        if data["xai"].get("spectral_shap"):
                            st.image(Image.open(io.BytesIO(base64.b64decode(data["xai"]["spectral_shap"]))), use_column_width=True)
                        else:
                            st.warning("SHAP visualization not available.")
                            
                    with xai_tabs[2]:
                        st.markdown("#### High-Frequency Spectral Distributions")
                        hm_cols = st.columns(3)
                        heatmaps = data["xai"].get("spectral_heatmaps", {})
                        if heatmaps.get("fft"): hm_cols[0].image(Image.open(io.BytesIO(base64.b64decode(heatmaps["fft"]))), caption="FFT Spectrum")
                        if heatmaps.get("dct"): hm_cols[1].image(Image.open(io.BytesIO(base64.b64decode(heatmaps["dct"]))), caption="DCT Sub-bands")
                        if heatmaps.get("wavelet_hh"): hm_cols[2].image(Image.open(io.BytesIO(base64.b64decode(heatmaps["wavelet_hh"]))), caption="Wavelet HH High-Frequency")
                            
                    st.markdown("---")
                    st.markdown("### Automated Forensic Metadata")
                    st.json(data["metadata"])
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")