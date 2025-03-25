import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import string
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from .extract_feature import extract_features
from .preprocessing import preprocess_image
from .segmentation import segment_captcha_with_projection

CHARACTERS = string.ascii_lowercase + string.digits
PROJECTION_THRESHOLD = 0.1
IMG_HEIGHT, IMG_WIDTH = 40, 30

# One-hot encoding
def to_categorical(y, num_classes):
    '''
    Converts a list of integers to one-hot encoding
    '''
    return F.one_hot(y, num_classes=num_classes).numpy()  # Convert to numpy

# Prepare data for CNN training
def prepare_training_data(folder_path):
    """
    Denoise and tokenize captcha image files in a folder into individual characters using vertical projection.
    Ignores image files where segmentation has failed (num_of_segmented_char =/= actual_num_of_char).

    Parameters:
    folder_path (str): Path to the folder containing CAPTCHA images. Image files in the folder should be of format "captchachars-0" （e.g "abc123-0")

    Returns:
    tuple：
    - X_img：Numpy array of Image of char with type numpy array (40 x 30)
    - feature_list: Dictionary of features extracted from the corresponding image 
    - y: Numpy array of one-hot encoded label of the corresponding image
    """
    all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    X_img = []  # Images
    X_features_list = []  # Engineered features
    y = []  # Labels

    
    for filename in tqdm(all_images, desc="Preparing Training Data"):
        image_path = os.path.join(folder_path, filename)
        filename_without_suffix = os.path.splitext(filename)[0]
        correct_characters = filename_without_suffix.split('-')[0]
        
        image = cv2.imread(image_path)
        thresh = preprocess_image(image)
        character_boundaries, _, _ = segment_captcha_with_projection(thresh, PROJECTION_THRESHOLD)
        
        # Skip if segmentation failed or number of segments doesn't match expected characters
        if len(character_boundaries) != len(correct_characters):
            continue
        
        for i, (start, end) in enumerate(character_boundaries):
            char_image = thresh[:, start:end]
            char_label = correct_characters[i]
            
            # Skip if character is not in our expected set
            if char_label not in CHARACTERS:
                continue
            
            # Extract features and resized image
            features, char_image_resized = extract_features(char_image)
            
            # Add to dataset
            X_img.append(char_image_resized)
            X_features_list.append(features)
            y.append(CHARACTERS.index(char_label))
            
    # Convert to numpy arrays
    X_img = np.array(X_img)

    # convert to tensor
    y = torch.tensor(y, dtype=torch.long)
    
    # Convert features to DataFrame and then to numpy array
    features_df = pd.DataFrame(X_features_list)
    X_features = features_df.values
    feature_names = list(features_df.columns)
    
    # Reshape for CNN input
    X_img = np.expand_dims(X_img, 1)
    X_img = X_img / 255.0  # Normalize
    
    # One-hot encode labels
    y_one_hot = to_categorical(y, num_classes=len(CHARACTERS))
    
    # Standardize features
    scaler = StandardScaler()
    X_features_scaled = None
    X_features_scaled = scaler.fit_transform(X_features)
    
    return X_img.astype(np.float32), X_features_scaled, y_one_hot, feature_names, scaler # Convert to float for pytorch