import os
import cv2
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.spectral_engine import SpectralAnalyzer

def train_model():
    print("=== 1. Initializing Feature Extraction Pipeline ===")
    analyzer = SpectralAnalyzer()
    
    real_dir = "data/dataset/real"
    fake_dir = "data/dataset/fake"
    
    data_rows = []
    
    # 1. Extract features from Real images (Label = 0)
    print("Extracting frequency metrics from Real faces...")
    for filename in os.listdir(real_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(real_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                features, _ = analyzer.extract_all(img)
                features["label"] = 0  # 0 = Authentic
                data_rows.append(features)
                
    # 2. Extract features from Fake images (Label = 1)
    print("Extracting frequency metrics from AI-Generated faces...")
    for filename in os.listdir(fake_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(fake_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                features, _ = analyzer.extract_all(img)
                features["label"] = 1  # 1 = Deepfake
                data_rows.append(features)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data_rows)
    print(f"\n -> Extraction Complete! Built feature matrix with shape: {df.shape}")

    # Separate input metrics (X) and target labels (y)
    X = df.drop(columns=["label"])
    y = df["label"]

    # Save feature names so we can use them during live inference later
    feature_names = list(X.columns)
    os.makedirs("models", exist_ok=True)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # 3. Split into 80% Training and 20% Testing sets
    print("\n=== 2. Splitting Data & Training XGBoost Classifier ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # Train model on CPU/Arm64 fast-path
    model.fit(X_train, y_train)
    print(" -> Model training complete!")

    # 4. Evaluate Performance on Unseen Test Set
    print("\n=== 3. Model Evaluation & Academic Metrics ===")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Deepfake (1)"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f" [ True Real: {cm[0][0]:2d}  |  False Alarm: {cm[0][1]:2d} ]")
    print(f" [ Missed   : {cm[1][0]:2d}  |  True Fake  : {cm[1][1]:2d} ]")

    # 5. Save the trained model
    model_path = "models/xgboost_spectral_model.json"
    model.save_model(model_path)
    print(f"\n -> Successfully saved production classifier to: {model_path}")
    print("\nStep 2 Complete! You now have a functional ML authentication model ready for live predictions.")

if __name__ == "__main__":
    train_model()