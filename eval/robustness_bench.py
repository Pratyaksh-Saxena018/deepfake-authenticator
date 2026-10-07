import os
import cv2
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import albumentations as A
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spectral_engine import SpectralAnalyzer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def apply_perturbation(image_np, perturbation_type):
    if perturbation_type == "jpeg_90":
        return A.ImageCompression(quality_lower=90, quality_upper=90, p=1.0)(image=image_np)['image']
    elif perturbation_type == "jpeg_75":
        return A.ImageCompression(quality_lower=75, quality_upper=75, p=1.0)(image=image_np)['image']
    elif perturbation_type == "jpeg_50":
        return A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0)(image=image_np)['image']
    elif perturbation_type == "jpeg_30":
        return A.ImageCompression(quality_lower=30, quality_upper=30, p=1.0)(image=image_np)['image']
    elif perturbation_type == "downscale_0.75":
        h, w = image_np.shape[:2]
        return A.Resize(int(h * 0.75), int(w * 0.75), p=1.0)(image=image_np)['image']
    elif perturbation_type == "downscale_0.5":
        h, w = image_np.shape[:2]
        return A.Resize(int(h * 0.5), int(w * 0.5), p=1.0)(image=image_np)['image']
    elif perturbation_type == "blur_gaussian":
        return A.GaussianBlur(blur_limit=(7, 7), p=1.0)(image=image_np)['image']
    elif perturbation_type == "noise_gaussian":
        return A.GaussNoise(var_limit=(50.0, 50.0), p=1.0)(image=image_np)['image']
    elif perturbation_type == "brightness_contrast":
        return A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)(image=image_np)['image']
    else:
        return image_np

def run_robustness_benchmark():
    print("=== Automated Robustness Stress-Test ===")
    real_dir = "../data/dataset/real" if os.path.exists("../data/dataset/real") else "data/dataset/real"
    fake_dir = "../data/dataset/fake" if os.path.exists("../data/dataset/fake") else "data/dataset/fake"
    
    file_paths, labels = [], []
    for d, l in [(real_dir, 0), (fake_dir, 1)]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_paths.append(os.path.join(d, f))
                    labels.append(l)
                    
    if not file_paths:
        print("No test data available.")
        return

    # Load models
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    spatial_model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    
    model_path_pt = "../models/mobilevit_deepfake.pt" if os.path.exists("../models/mobilevit_deepfake.pt") else "models/mobilevit_deepfake.pt"
    if os.path.exists(model_path_pt):
        spatial_model.load_state_dict(torch.load(model_path_pt, map_location=device))
    spatial_model.eval()

    xgb_model = xgb.XGBClassifier()
    model_path_xgb = "../models/xgboost_spectral_model.json" if os.path.exists("../models/xgboost_spectral_model.json") else "models/xgboost_spectral_model.json"
    if os.path.exists(model_path_xgb):
        xgb_model.load_model(model_path_xgb)
    
    analyzer = SpectralAnalyzer()

    perturbations = [
        "clean", "jpeg_90", "jpeg_75", "jpeg_50", "jpeg_30", 
        "downscale_0.75", "downscale_0.5", "blur_gaussian", 
        "noise_gaussian", "brightness_contrast"
    ]
    
    results = []

    for pert in perturbations:
        print(f"\nRunning evaluate on: {pert}")
        preds_spatial = []
        preds_spectral = []
        preds_ensemble = []
        
        for idx, path in enumerate(file_paths):
            image = Image.open(path).convert("RGB")
            image_np = np.array(image)
            
            if pert != "clean":
                image_np = apply_perturbation(image_np, pert)
                image = Image.fromarray(image_np)
            
            # Spatial Prediction
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = spatial_model(pixel_values=inputs["pixel_values"].to(device)).logits
                sp_prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            # Spectral Prediction
            # Need to process as BGR since SpectralAnalyzer takes BGR (from cv2.imread)
            img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            features, _ = analyzer.extract_all(img_bgr)
            df_features = pd.DataFrame([features])
            
            # If XGBoost is trained, we use it. If not, mock 0.5
            if os.path.exists(model_path_xgb):
                # Ensure feature order matches
                spec_prob = xgb_model.predict_proba(df_features)[0, 1]
            else:
                spec_prob = 0.5
                
            ens_prob = (sp_prob + spec_prob) / 2.0
            
            preds_spatial.append(sp_prob)
            preds_spectral.append(spec_prob)
            preds_ensemble.append(ens_prob)

        y_true = np.array(labels)
        roc = roc_auc_score(y_true, preds_ensemble)
        acc = accuracy_score(y_true, np.array(preds_ensemble) > 0.5)
        
        results.append({
            "Perturbation": pert,
            "Ensemble AUC": round(roc, 4),
            "Ensemble Acc": round(acc, 4)
        })
        print(f" -> AUC: {roc:.4f} | Acc: {acc:.4f}")

    print("\n=== Robustness Summary ===")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_robustness_benchmark()
