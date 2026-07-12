import os
import shutil
import cv2
import numpy as np
from datasets import load_dataset
from utils.face_extractor import MediaExtractor

def build_dataset(sample_size_per_class=200):
    print("=== 1. Cleaning Old Data & Creating Directories ===")
    real_dir = "data/dataset/real"
    fake_dir = "data/dataset/fake"
    
    # Clean out any previously generated grey square grids
    if os.path.exists("data/dataset"):
        shutil.rmtree("data/dataset")
        
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    extractor = MediaExtractor(margin=0.20)

    print(f"=== 2. Fetching Real Benchmark Faces from Hugging Face ===")
    print("Connecting to public dataset 'Hemg/deepfake-and-real-images'...")
    
    try:
        # We use streaming=True so we don't download the entire 190,000 image dataset!
        dataset = load_dataset("Hemg/deepfake-and-real-images", split="train", streaming=True)
        
        real_count = 0
        fake_count = 0
        
        for item in dataset:
            if real_count >= sample_size_per_class and fake_count >= sample_size_per_class:
                break
                
            # Convert PIL image to BGR OpenCV format
            pil_img = item['image'].convert('RGB')
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            label = item['label']  # In this dataset: 1 = Real, 0 = Fake
            
            # Use our built-in OpenCV Haar Cascade to crop the face
            face = extractor.extract_face(img_bgr)
            if face is None:
                # If the face detector misses, resize the center image
                face = cv2.resize(img_bgr, (256, 256))
                
            # Sort into Real vs Fake folders
            if (label == 1 or label == 'Real' or label == 'real') and real_count < sample_size_per_class:
                cv2.imwrite(f"{real_dir}/real_{real_count:03d}.jpg", face)
                real_count += 1
                if real_count % 50 == 0:
                    print(f" -> Downloaded & Cropped {real_count}/{sample_size_per_class} Authentic Human Faces...")
                    
            elif (label == 0 or label == 'Fake' or label == 'fake') and fake_count < sample_size_per_class:
                cv2.imwrite(f"{fake_dir}/fake_{fake_count:03d}.jpg", face)
                fake_count += 1
                if fake_count % 50 == 0:
                    print(f" -> Downloaded & Cropped {fake_count}/{sample_size_per_class} AI-Generated Faces...")

    except Exception as e:
        print(f"\n[Error] Could not connect to Hugging Face Hub: {e}")
        return

    print("\n=== Dataset Preparation Complete! ===")
    print(f"Authentic human faces saved to : {real_dir} ({len(os.listdir(real_dir))} images)")
    print(f"AI-Generated faces saved to    : {fake_dir} ({len(os.listdir(fake_dir))} images)")
    print("\n -> Open your data/dataset folder now to verify real human faces!")

if __name__ == "__main__":
    build_dataset(sample_size_per_class=200)