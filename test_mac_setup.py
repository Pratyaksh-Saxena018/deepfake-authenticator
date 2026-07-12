import platform
import torch
import cv2
import mediapipe as mp
import xgboost as xgb

print("=== System Architecture Check ===")
print(f"Processor Architecture : {platform.machine()}")
print(f"Python Version         : {platform.python_version()}")

print("\n=== Hardware Acceleration Check ===")
mps_available = torch.backends.mps.is_available()
print(f"PyTorch Metal (MPS) GPU Available : {mps_available}")

if mps_available:
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(" -> Success: PyTorch can execute tensor math natively on your Mac GPU!")
else:
    print(" -> Notice: Running on CPU (Still extremely fast on Apple Silicon)")

print("\n=== Core Libraries Check ===")
print(f"OpenCV Version    : {cv2.__version__}")
print(f"XGBoost Version   : {xgb.__version__}")
print(" -> All media and machine learning libraries loaded successfully!")
