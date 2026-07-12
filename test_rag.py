import cv2
import xgboost as xgb
import pandas as pd
from utils.spectral_engine import SpectralAnalyzer
from utils.rag_reporter import ForensicReporter

def test_reporting():
    print("=== 1. Loading Trained XGBoost Classifier ===")
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_spectral_model.json")
    
    print("=== 2. Analyzing Sample Image ===")
    # Grab the very first AI-generated face from your downloaded dataset
    sample_path = "data/dataset/fake/fake_000.jpg"
    img = cv2.imread(sample_path)
    
    if img is None:
        print(" -> Error: Could not load test image. Make sure data/dataset/fake/fake_000.jpg exists!")
        return

    analyzer = SpectralAnalyzer()
    features, heatmaps = analyzer.extract_all(img)
    
    # Format features into a 1-row DataFrame matching the XGBoost training schema
    df_features = pd.DataFrame([features])
    
    # Get probability score (Index 1 corresponds to Class 1: Fake)
    fake_prob = float(model.predict_proba(df_features)[0][1])
    
    print(f" -> XGBoost Deepfake Probability: {fake_prob * 100:.2f}%\n")
    
    print("=== 3. Executing RAG Forensic Reporter ===")
    # Running without an API key to evaluate the structured local engine fallback
    reporter = ForensicReporter() 
    report_markdown = reporter.generate_report(features, fake_prob)
    
    print("\n" + "="*70)
    print(report_markdown)
    print("="*70)
    print("\nStep 3 Complete! Your RAG forensic reporting engine is operational.")

if __name__ == "__main__":
    test_reporting()