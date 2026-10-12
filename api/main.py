import os
import io
import base64
import json
import numpy as np
import cv2
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
import xgboost as xgb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import shap
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spectral_engine import SpectralAnalyzer

app = FastAPI(title="Deepfake Spectral Authenticator API", version="1.1")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
spatial_model = None
processor = None
spectral_model = None
meta_model = None
analyzer = None
explainer = None

@app.on_event("startup")
async def load_models():
    global spatial_model, processor, spectral_model, meta_model, analyzer, explainer
    
    print("Loading models...")
    # 1. Spatial Model
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    spatial_model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
    )
    if os.path.exists("models/mobilevit_deepfake.pt"):
        spatial_model.load_state_dict(torch.load("models/mobilevit_deepfake.pt", map_location=device))
    spatial_model.to(device)
    spatial_model.eval()

    # 2. Spectral Model & SHAP
    spectral_model = xgb.XGBClassifier()
    if os.path.exists("models/xgboost_spectral_model.json"):
        spectral_model.load_model("models/xgboost_spectral_model.json")
        explainer = shap.TreeExplainer(spectral_model)

    # 3. Meta-Classifier (Smart Ensemble)
    if os.path.exists("models/meta_classifier.pkl"):
        meta_model = joblib.load("models/meta_classifier.pkl")
        print("Meta-Classifier loaded successfully.")

    analyzer = SpectralAnalyzer()
    print("All models loaded.")

def get_base64_image(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def generate_gradcam(image_tensor, image_np):
    try:
        # CRITICAL FIX: GradCAM requires gradients. We must temporarily enable them.
        with torch.enable_grad():
            tensor_for_cam = image_tensor.clone().requires_grad_(True)
            target_layers = [spatial_model.mobilevit.conv_1x1_exp]
            cam = GradCAM(model=spatial_model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(1)]
            
            grayscale_cam = cam(input_tensor=tensor_for_cam, targets=targets)[0, :]
            img_normalized = image_np.astype(np.float32) / 255.0
            cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
            
            return get_base64_image(cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"GradCAM error: {e}")
        return None

def generate_shap_plot(features_df):
    try:
        if explainer is None: return None
        shap_values = explainer.shap_values(features_df)
        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value, 
                                             data=features_df.iloc[0], 
                                             feature_names=features_df.columns), 
                            show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return get_base64_image(cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR))
    except Exception as e:
        print(f"SHAP error: {e}")
        return None

@app.post("/api/v1/authenticate")
async def authenticate(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np, image_bgr = np.array(image), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Spatial Inference
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        spatial_prob = torch.softmax(spatial_model(pixel_values=pixel_values).logits, dim=1)[0, 1].item()

    # XAI Spatial
    gradcam_b64 = generate_gradcam(pixel_values, cv2.resize(image_np, (256, 256)))

    # Spectral Inference & XAI
    features, heatmaps = analyzer.extract_all(image_bgr)
    df_features = pd.DataFrame([features])
    spectral_prob, shap_b64 = 0.5, None
    if spectral_model is not None:
        spectral_prob = spectral_model.predict_proba(df_features)[0, 1]
        shap_b64 = generate_shap_plot(df_features)

    # Heatmaps formatting
    def norm_hm(hm):
        return get_base64_image(cv2.applyColorMap(cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))

    # Smart Meta-Classifier Logic
    if meta_model is not None:
        ensemble_prob = float(meta_model.predict_proba([[spatial_prob, spectral_prob]])[0, 1])
    else:
        ensemble_prob = float((spatial_prob + spectral_prob) / 2.0)

    verdict = "Deepfake" if ensemble_prob > 0.5 else "Authentic"
    return JSONResponse(content={
        "spatial_score": float(spatial_prob),
        "spectral_score": float(spectral_prob),
        "ensemble_score": ensemble_prob,
        "verdict": verdict,
        "confidence": ensemble_prob if verdict == "Deepfake" else 1.0 - ensemble_prob,
        "metadata": {"fft_slope": features["fft_slope"], "dct_energy_ratio": features["dct_energy_ratio"], "wavelet_energy": features["wavelet_energy"]},
        "xai": {"spatial_gradcam": gradcam_b64, "spectral_shap": shap_b64, "spectral_heatmaps": {"fft": norm_hm(heatmaps["fft"]), "dct": norm_hm(heatmaps["dct"]), "wavelet_hh": norm_hm(heatmaps["wavelet_hh"])}}
    })