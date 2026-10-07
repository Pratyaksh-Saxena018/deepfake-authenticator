import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score, confusion_matrix, classification_report
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
import albumentations as A
import sys

# Add parent to path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spectral_engine import SpectralAnalyzer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def calculate_sensitivity_at_far(y_true, y_prob, far_target=0.01):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if len(fpr) <= 1:
        return 0.0
    idx = np.where(fpr <= far_target)[0][-1]
    return tpr[idx]

class EvalDataset(Dataset):
    def __init__(self, file_paths, labels, processor, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = Image.fromarray(augmented['image'])
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def run_kfold_evaluation():
    print("=== Starting Enterprise 4-Fold Validation (Pooled OOF) ===")
    real_dir = "../data/dataset/real" if os.path.exists("../data/dataset/real") else "data/dataset/real"
    fake_dir = "../data/dataset/fake" if os.path.exists("../data/dataset/fake") else "data/dataset/fake"
    
    file_paths = []
    labels = []
    
    for d, l in [(real_dir, 0), (fake_dir, 1)]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_paths.append(os.path.join(d, f))
                    labels.append(l)

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    if len(file_paths) == 0:
        print("No data found for evaluation!")
        return

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_probs_ensemble = np.zeros(len(file_paths))
    
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    analyzer = SpectralAnalyzer()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n--- FOLD {fold + 1}/4 ---")
        train_paths, val_paths = file_paths[train_idx], file_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # 1. Train MobileViT
        print("Training Spatial Model (MobileViT)...")
        train_ds = EvalDataset(train_paths, train_labels, processor)
        val_ds = EvalDataset(val_paths, val_labels, processor)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        
        spatial_model = MobileViTForImageClassification.from_pretrained(
            "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
        ).to(device)
        
        optimizer = torch.optim.AdamW(spatial_model.parameters(), lr=3e-5)
        epochs = 3 # Reduced epochs for CV speed
        for epoch in range(epochs):
            spatial_model.train()
            for images, tgts in train_loader:
                images, tgts = images.to(device), tgts.to(device)
                optimizer.zero_grad()
                outputs = spatial_model(pixel_values=images).logits
                loss = nn.CrossEntropyLoss()(outputs, tgts)
                loss.backward()
                optimizer.step()
        
        # 2. Extract Spectral Features for XGBoost
        print("Training Spectral Model (XGBoost)...")
        def extract_features(paths):
            feats = []
            for p in paths:
                img = cv2.imread(p)
                f, _ = analyzer.extract_all(img)
                feats.append(f)
            return pd.DataFrame(feats)
        
        X_train = extract_features(train_paths)
        X_val = extract_features(val_paths)
        
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
        xgb_model.fit(X_train, train_labels)
        
        # 3. Predict on Validation Fold (Ensemble)
        spatial_model.eval()
        spatial_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                logits = spatial_model(pixel_values=images).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                spatial_probs.extend(probs)
        spatial_probs = np.array(spatial_probs)
        
        spectral_probs = xgb_model.predict_proba(X_val)[:, 1]
        
        # Ensemble Probability (Average)
        ensemble_probs = (spatial_probs + spectral_probs) / 2.0
        oof_probs_ensemble[val_idx] = ensemble_probs

    print("\n=== Pooled K4 Validation Results ===")
    y_pred = (oof_probs_ensemble > 0.5).astype(int)
    
    roc_auc = roc_auc_score(labels, oof_probs_ensemble)
    ll = log_loss(labels, oof_probs_ensemble)
    brier = brier_score_loss(labels, oof_probs_ensemble)
    macro_f1 = f1_score(labels, y_pred, average="macro")
    sens_1_far = calculate_sensitivity_at_far(labels, oof_probs_ensemble, 0.01)
    
    print(f"OOF ROC-AUC     : {roc_auc:.4f}")
    print(f"LogLoss         : {ll:.4f}")
    print(f"Brier Score     : {brier:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Sens @ 1% FAR   : {sens_1_far:.4f}")
    
    print("\nCombined Confusion Matrix:")
    print(confusion_matrix(labels, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(labels, y_pred, target_names=["Authentic (0)", "Deepfake (1)"]))

if __name__ == "__main__":
    run_kfold_evaluation()
