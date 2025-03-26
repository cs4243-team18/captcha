import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helper_functions.encode import (
    CHARACTERS, 
    segment_captcha_with_projection, preprocess_image,
    PROJECTION_THRESHOLD, IMG_HEIGHT, IMG_WIDTH
)
import cv2
import os
from tqdm import tqdm

# Custom Dataset Class
class CharDataset(Dataset):
    def __init__(self, X_img, y):
        # Input is already in NCHW format (N,1,40,30)
        self.X = torch.tensor(X_img, dtype=torch.float32)  # Remove .permute()
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CNN Model Architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 7, 128)  # Adjusted for 40x30 input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 10 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Character-Level Evaluation Function
def evaluate_character_level(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }

# Captcha-Level Evaluation Function
def evaluate_captcha_level(model, folder_path, device):
    model.eval()
    correct = 0
    total = 0
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(all_images, desc="Evaluating CAPTCHAs"):
        image_path = os.path.join(folder_path, filename)
        correct_label = os.path.splitext(filename)[0].split('-')[0]
        
        # Preprocess and segment
        image = cv2.imread(image_path)
        thresh = preprocess_image(image)
        boundaries, _, _ = segment_captcha_with_projection(thresh, PROJECTION_THRESHOLD)
        
        if len(boundaries) != len(correct_label):
            total += 1
            continue
        
        # Process each character
        predicted_chars = []
        for start, end in boundaries:
            char_img = thresh[:, start:end]
            resized = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))
            resized = resized.reshape(IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
            
            # Convert to tensor and predict
            tensor_img = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor_img)
                _, pred = torch.max(outputs, 1)
                predicted_chars.append(CHARACTERS[pred.item()])
        
        # Check full CAPTCHA
        if ''.join(predicted_chars) == correct_label:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0