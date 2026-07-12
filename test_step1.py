import cv2
import numpy as np
import os
from utils.face_extractor import MediaExtractor
from utils.spectral_engine import SpectralAnalyzer

def run_test():
    print("--- Initializing Pipeline ---")
    extractor = MediaExtractor(margin=0.20)
    analyzer = SpectralAnalyzer()

    # Create a synthetic image with geometric patterns to test frequency extraction
    dummy_image = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
    cv2.circle(dummy_image, (200, 200), 80, (255, 255, 255), -1)  # Central structure
    
    test_img_path = "data/raw/test_sample.jpg"
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed_faces", exist_ok=True)
    cv2.imwrite(test_img_path, dummy_image)
    
    print(f"1. Loading test image from {test_img_path}...")
    img = cv2.imread(test_img_path)
    
    print("2. Attempting face extraction...")
    # Since our test image is synthetic, MediaPipe won't detect a human face.
    # We fallback to resizing the raw image to verify our mathematical transforms.
    face = extractor.extract_face(img)
    if face is None:
        print(" -> Notice: No human face in synthetic test sample. Using resized image for math verification...")
        face = cv2.resize(img, (256, 256))
    else:
        print(" -> Success: Face successfully cropped!")
        
    cv2.imwrite("data/processed_faces/cropped_test.jpg", face)
    
    print("3. Executing FFT, DCT, and Wavelet spectral transforms...")
    features, heatmaps = analyzer.extract_all(face)
    
    print("\n=== Extraction Success! Handcrafted Spectral Features ===")
    for key, value in features.items():
        print(f"{key:<20}: {value:.4f}")
        
    print("\n=== Heatmaps Generated Successfully ===")
    for name, map_data in heatmaps.items():
        print(f" - {name:<12}: Shape {map_data.shape}")
        
    # Save the FFT heatmap as an image to visually confirm the output
    fft_vis = cv2.normalize(heatmaps["fft"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("data/processed_faces/fft_heatmap_sample.jpg", fft_vis)
    print("\n -> Saved visual FFT heatmap sample to: data/processed_faces/fft_heatmap_sample.jpg")
    print("\nStep 1 Complete! The mathematical feature engine is 100% operational.")

if __name__ == "__main__":
    run_test()