import os
import io
import base64
import json
import numpy as np
import cv2
import pandas as pd
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

app = FastAPI(title="Deepfake Spectral Authenticator API", version="1.0")

# Setup device and globals
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
spatial_model = None
processor = None
spectral_model = None
analyzer = None
explainer = None

@app.on_event("startup")
async def load_models():
    global spatial_model, processor, spectral_model, analyzer, explainer
    
    print("Loading models...")
    # 1. Load Spatial Model (MobileViT)
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    spatial_model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
    )
    model_path_pt = "models/mobilevit_deepfake.pt"
    if os.path.exists(model_path_pt):
        spatial_model.load_state_dict(torch.load(model_path_pt, map_location=device))
    spatial_model.to(device)
    spatial_model.eval()

    # 2. Load Spectral Model (XGBoost)
    spectral_model = xgb.XGBClassifier()
    model_path_xgb = "models/xgboost_spectral_model.json"
    if os.path.exists(model_path_xgb):
        spectral_model.load_model(model_path_xgb)
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(spectral_model)

    analyzer = SpectralAnalyzer()
    print("Models loaded successfully.")

def get_base64_image(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def generate_gradcam(image_tensor, image_np):
    try:
        # Target layer for MobileViT (last conv layer before classifier)
        target_layers = [spatial_model.mobilevit.conv_1x1_exp]
        cam = GradCAM(model=spatial_model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)] # Target class 1 (Deepfake)
        
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        
        img_normalized = image_np.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        # Convert RGB to BGR for opencv encoding
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        return get_base64_image(cam_image_bgr)
    except Exception as e:
        print(f"GradCAM error: {e}")
        return None

def generate_shap_plot(features_df):
    try:
        if explainer is None:
            return None
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
        
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return get_base64_image(img)
    except Exception as e:
        print(f"SHAP error: {e}")
        return None

@app.post("/api/v1/authenticate")
async def authenticate(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 1. Spatial Inference
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    with torch.no_grad():
        logits = spatial_model(pixel_values=pixel_values).logits
        spatial_prob = torch.softmax(logits, dim=1)[0, 1].item()

    # Generate Grad-CAM Heatmap
    gradcam_b64 = generate_gradcam(pixel_values, cv2.resize(image_np, (256, 256)))

    # 2. Spectral Inference
    features, heatmaps = analyzer.extract_all(image_bgr)
    df_features = pd.DataFrame([features])
    
    # Need to match feature names if model is loaded
    spectral_prob = 0.5
    shap_b64 = None
    if spectral_model is not None:
        try:
            spectral_prob = spectral_model.predict_proba(df_features)[0, 1]
            shap_b64 = generate_shap_plot(df_features)
        except Exception as e:
            print(f"XGBoost Inference error: {e}")

    # Process heatmaps for base64
    def normalize_heatmap(hm):
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
        return get_base64_image(hm_color)
        
    spectral_heatmaps_b64 = {
        "fft": normalize_heatmap(heatmaps["fft"]),
        "dct": normalize_heatmap(heatmaps["dct"]),
        "wavelet_hh": normalize_heatmap(heatmaps["wavelet_hh"])
    }

    # 3. Ensemble
    ensemble_prob = float((spatial_prob + spectral_prob) / 2.0)
    verdict = "Deepfake" if ensemble_prob > 0.5 else "Authentic"
    confidence = float(ensemble_prob if verdict == "Deepfake" else 1.0 - ensemble_prob)

    return JSONResponse(content={
        "spatial_score": float(spatial_prob),
        "spectral_score": float(spectral_prob),
        "ensemble_score": ensemble_prob,
        "verdict": verdict,
        "confidence": confidence,
        "metadata": {
            "fft_slope": features["fft_slope"],
            "dct_energy_ratio": features["dct_energy_ratio"],
            "wavelet_energy": features["wavelet_energy"]
        },
        "xai": {
            "spatial_gradcam": gradcam_b64,
            "spectral_shap": shap_b64,
            "spectral_heatmaps": spectral_heatmaps_b64
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
