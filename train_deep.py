import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from sklearn.model_selection import train_test_split
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Optimized Training Platform Active: {device}")

# 1. Standardizing Augmentation Pipeline to destroy memorization
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ImageCompression(quality_lower=30, quality_upper=90, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(p=0.5)
])

class CIPLabDataset(Dataset):
    def __init__(self, file_paths, labels, processor, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert("RGB")
            
            # Apply dynamic geometrical variation only during training
            if self.transform:
                image_np = np.array(image)
                augmented = self.transform(image=image_np)
                image = Image.fromarray(augmented['image'])
                
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            return torch.zeros(3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.long)

def gather_paths():
    real_dir = "data/dataset/real"
    fake_dir = "data/dataset/fake"
    file_paths, labels = [], []

    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(real_dir, f))
                labels.append(0)

    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(fake_dir, f))
                labels.append(1)
        
    return train_test_split(file_paths, labels, test_size=0.2, random_state=42)

def execute_training():
    train_paths, val_paths, train_labels, val_labels = gather_paths()
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    
    # Inject transforms into training set, leave validation set clean
    train_ds = CIPLabDataset(train_paths, train_labels, processor, transform=train_transform)
    val_ds = CIPLabDataset(val_paths, val_labels, processor, transform=None)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Injected weight_decay=1e-3 to penalize over-complicated memorized boundaries
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-3)
    
    # Injected a Learning Rate Scheduler: cuts learning rate when progress stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = 12  # Dropped from 15 since scheduler/augmentations converge smarter
    best_val_loss = float('inf')
    
    print("🏋️ Starting Anti-Overfitting Optimization Cycle...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(pixel_values=val_images).logits
                v_loss = criterion(val_outputs, val_labels)
                
                val_loss += v_loss.item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler based on validation loss progression
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save only the best weights based on true generalizability (lowest val loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/mobilevit_deepfake.pt")
            print("💾 Better generalization target located. Updated weights saved.")

    print(f"✅ Training complete. Best generalization profile achieved with Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    execute_training()