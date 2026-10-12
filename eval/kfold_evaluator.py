import os
# --- ANTI-SEGFAULT THREAD LOCKS ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import cv2
import torch
torch.set_num_threads(1) # Prevent PyTorch from clashing with XGBoost
device = torch.device("cpu") # Strictly force CPU to bypass MPS memory leaks

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, f1_score, confusion_matrix, classification_report
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spectral_engine import SpectralAnalyzer

def calculate_sensitivity_at_far(y_true, y_prob, far_target=0.01):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if len(fpr) <= 1: return 0.0
    return tpr[np.where(fpr <= far_target)[0][-1]]

class EvalDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths, self.labels, self.processor = file_paths, labels, processor
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        inputs = self.processor(images=Image.open(self.file_paths[idx]).convert("RGB"), return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

def run_kfold_evaluation():
    print("=== Starting Enterprise 4-Fold Validation & Meta-Classifier Training ===")
    real_dir = "data/dataset/real" if os.path.exists("data/dataset/real") else "../data/dataset/real"
    fake_dir = "data/dataset/fake" if os.path.exists("data/dataset/fake") else "../data/dataset/fake"
    
    # 1. Load File Paths
    file_paths, labels = [], []
    for d, l in [(real_dir, 0), (fake_dir, 1)]:
        if os.path.exists(d):
            file_paths.extend([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))])
            labels.extend([l] * len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))]))

    file_paths, labels = np.array(file_paths), np.array(labels)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    oof_spatial, oof_spectral = np.zeros(len(file_paths)), np.zeros(len(file_paths))
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    analyzer = SpectralAnalyzer()
    
    # 2. Loop Through Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n--- FOLD {fold + 1}/4 ---")
        train_paths, val_paths, train_labels, val_labels = file_paths[train_idx], file_paths[val_idx], labels[train_idx], labels[val_idx]
        
        # 3. Spatial Inference (IMPROVED: Head-Only Transfer Learning)
        spatial_model = MobileViTForImageClassification.from_pretrained(
            "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
        ).to(device)
        
        # Freeze the heavy Transformer backbone
        for param in spatial_model.mobilevit.parameters():
            param.requires_grad = False
            
        train_loader = DataLoader(EvalDataset(train_paths, train_labels, processor), batch_size=16, shuffle=True)
        val_loader = DataLoader(EvalDataset(val_paths, val_labels, processor), batch_size=16, shuffle=False)
        
        # Only optimize the final classifier head
        optimizer = torch.optim.AdamW(spatial_model.classifier.parameters(), lr=1e-3)
        
        print("Training Spatial Model (Head-Only)...")
        for epoch in range(5): 
            spatial_model.train()
            for images, tgts in train_loader:
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(spatial_model(pixel_values=images.to(device)).logits, tgts.to(device)).backward()
                optimizer.step()
        
        # Evaluate Spatial Model
        spatial_model.eval()
        spatial_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                spatial_probs.extend(torch.softmax(spatial_model(pixel_values=images.to(device)).logits, dim=1)[:, 1].cpu().numpy())
        oof_spatial[val_idx] = np.array(spatial_probs)
        
        # 4. Spectral Inference (XGBoost)
        print("Training Spectral Model (XGBoost)...")
        X_train = pd.DataFrame([analyzer.extract_all(cv2.imread(p))[0] for p in train_paths])
        X_val = pd.DataFrame([analyzer.extract_all(cv2.imread(p))[0] for p in val_paths])
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=1).fit(X_train, train_labels)
        oof_spectral[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # 5. Train the Meta-Classifier
    print("\nTraining Logistic Regression Meta-Classifier...")
    X_meta = np.column_stack((oof_spatial, oof_spectral))
    meta_model = LogisticRegression(n_jobs=1).fit(X_meta, labels)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(meta_model, "models/meta_classifier.pkl")
    print("✅ Meta-Classifier saved to models/meta_classifier.pkl")

    oof_meta_probs = meta_model.predict_proba(X_meta)[:, 1]
    y_pred = (oof_meta_probs > 0.5).astype(int)
    
    print("\n=== Pooled K4 Validation Results (Meta-Classifier) ===")
    print(f"OOF ROC-AUC     : {roc_auc_score(labels, oof_meta_probs):.4f}")
    print(f"LogLoss         : {log_loss(labels, oof_meta_probs):.4f}")
    print(f"Brier Score     : {brier_score_loss(labels, oof_meta_probs):.4f}")
    print(f"Macro F1        : {f1_score(labels, y_pred, average='macro'):.4f}")
    print(f"Sens @ 1% FAR   : {calculate_sensitivity_at_far(labels, oof_meta_probs, 0.01):.4f}")
    print("\nCombined Confusion Matrix:\n", confusion_matrix(labels, y_pred))

if __name__ == "__main__":
    run_kfold_evaluation()