import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from helper_functions.data_transformation import (
    CHARACTERS,
    segment_by_projection_v2,
    PROJECTION_THRESHOLD, IMG_HEIGHT, IMG_WIDTH
)
from helper_functions.preprocessing import preprocess_image_v1
from helper_functions.extract_features import extract_features
import cv2
import os
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F


class HybridDataset(Dataset):
    def __init__(self, X_img, X_features, y):
        self.X_img = torch.tensor(X_img, dtype=torch.float32)
        self.X_features = torch.tensor(X_features, dtype=torch.float32)
        self.y = y

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        return (self.X_img[idx], self.X_features[idx]), self.y[idx]


class HybridCNN(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Concatenated features
        self.fc1 = nn.Linear(64*10*7 + num_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_img, x_features):
        # Image processing
        x = self.pool(F.relu(self.conv1(x_img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*10*7)

        # Feature concatenation
        x_combined = torch.cat((x, x_features), dim=1)
        x_combined = F.relu(self.fc1(x_combined))
        x_combined = self.fc2(x_combined)
        return x_combined

# Character-Level Evaluation Function
def evaluate_character_level(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Correctly unpack the nested structure
            (inputs_img, inputs_features), labels = batch
            
            # Move to device
            inputs_img = inputs_img.to(device)
            inputs_features = inputs_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs_img, inputs_features)
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
def evaluate_captcha_level(model, folder_path, device, scaler, feature_names):
    model.eval()
    correct = 0
    total = 0
    all_images = [f for f in os.listdir(
        folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(all_images, desc="Evaluating CAPTCHAs"):
        image_path = os.path.join(folder_path, filename)
        correct_label = os.path.splitext(filename)[0].split('-')[0]

        image = cv2.imread(image_path)
        thresh = preprocess_image_v1(image)
        boundaries, _, _ = segment_by_projection_v2(
            thresh, PROJECTION_THRESHOLD)

        if len(boundaries) != len(correct_label):
            total += 1
            continue

        predicted_chars = []
        for (start, end), true_char in zip(boundaries, correct_label):
            char_img = thresh[:, start:end]

            # Extract and scale features
            features, char_image_resized = extract_features(char_img)
            features_df = pd.DataFrame([features], columns=feature_names)
            X_features = scaler.transform(features_df.values)
            features_tensor = torch.tensor(
                X_features, dtype=torch.float32).to(device)

            # Prepare image tensor
            resized = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))
            resized = resized.reshape(1, 1, IMG_HEIGHT, IMG_WIDTH) / 255.0
            image_tensor = torch.tensor(
                resized, dtype=torch.float32).to(device)

            # Predict
            with torch.no_grad():
                # Add batch dimension to features if needed
                if len(features_tensor.shape) == 1:
                    features_tensor = features_tensor.unsqueeze(0)
                outputs = model(image_tensor, features_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_chars.append(CHARACTERS[pred.item()])

        # Check full CAPTCHA
        if ''.join(predicted_chars) == correct_label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
