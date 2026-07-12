import os
import sys
import cv2
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from utils.face_extractor import MediaExtractor
from utils.spectral_engine import SpectralAnalyzer

def quick_predict(image_path):
    # 1. Load the target image safely
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: System cannot find or open file at '{image_path}'")
        sys.exit(1)

    # 2. Isolate the face boundaries
    extractor = MediaExtractor(margin=0.20)
    face = extractor.extract_face(img)
    if face is None:
        face = cv2.resize(img, (256, 256)) # Fallback if no clean face geometry is found

    # 3. Transform into frequency metrics
    analyzer = SpectralAnalyzer()
    features, _ = analyzer.extract_all(face)

    # 4. Load weights and run mathematical inference
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_spectral_model.json")
    
    df_features = pd.DataFrame([features])
    fake_probability = float(model.predict_proba(df_features)[0][1])
    
    # 5. Output Minimal Terminal Results
    print("\n" + "="*45)
    print("      🔍 SPECTRUM FORENSIC AUTHENTICATOR")
    print("="*45)
    if fake_probability > 0.50:
        print(f" 🔴 VERDICT    : AI-GENERATED (DEEPFAKE)")
        print(f" 📊 CONFIDENCE : {fake_probability * 100:.2f}%")
    else:
        print(f" 🟢 VERDICT    : AUTHENTIC PHOTOGRAPH (CLICKED)")
        print(f" 📊 CONFIDENCE : {(1 - fake_probability) * 100:.2f}%")
    print("="*45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        quick_predict(sys.argv[1])
    else:
        path = input("Drag and drop your target image file here and hit Enter:\n> ").strip()
        # Clean terminal escape formatting added by Mac drag-and-drop actions
        path = path.replace("\\", "").replace("'", "").replace('"', "")
        quick_predict(path)