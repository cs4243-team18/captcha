import numpy as np
import cv2

from sklearn.preprocessing import StandardScaler

"""
Extract feature vector
"""
NUM_ENGINEERED_FEATURES = 9

def extract_features(char_image: np.ndarray) -> dict:
    # Extract features
    features = {}
    
    # 1. Aspect ratio
    height, width = char_image.shape
    features['aspect_ratio'] = width / height if height > 0 else 0
    
    # 2. Pixel density (ratio of white pixels to total pixels)
    features['pixel_density'] = np.sum(char_image > 0) / (height * width) if height * width > 0 else 0
    
    # 3. Horizontal symmetry
    flipped_h = cv2.flip(char_image, 1)
    features['h_symmetry'] = np.sum(char_image == flipped_h) / (height * width) if height * width > 0 else 0
    
    # 4. Vertical symmetry
    flipped_v = cv2.flip(char_image, 0)
    features['v_symmetry'] = np.sum(char_image == flipped_v) / (height * width) if height * width > 0 else 0
    
    # 5. Number of contours (complexity)
    contours, _ = cv2.findContours(char_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features['contour_count'] = len(contours)
    
    # 6. Horizontal and vertical projections
    h_proj = np.sum(char_image, axis=1) / width if width > 0 else np.zeros(height)
    v_proj = np.sum(char_image, axis=0) / height if height > 0 else np.zeros(width)
    
    features['h_proj_std'] = np.std(h_proj)
    features['v_proj_std'] = np.std(v_proj)
    
    # 7. Center of mass
    if np.sum(char_image) > 0:
        y_indices, x_indices = np.where(char_image > 0)
        features['com_x'] = np.mean(x_indices) / width if width > 0 else 0
        features['com_y'] = np.mean(y_indices) / height if height > 0 else 0
    else:
        features['com_x'] = 0.5
        features['com_y'] = 0.5
    
    return features