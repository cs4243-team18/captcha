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

from .extract_features import extract_features
from .preprocessing import preprocess_image_v1
from .segmentation import segment_by_projection_v1, segment_by_projection_v1_with_padding, segment_by_projection_v2

"""
Data Transformation methods which, after preprocessing and segmenting with methods from Phase 1 and 2 of the 
pipeline, obtains the transformed (X,y) data for Phase 3 (training the CNN model to classify characters).

In this project,
- X_imgs denotes the list of character images, each of shape (40, 30)
- y denotes the corresponding one-hot encoded labels
"""


CHARACTERS = string.digits + string.ascii_lowercase
PROJECTION_THRESHOLD = 0.1
IMG_HEIGHT = 40
IMG_WIDTH = 30


"""
[Private] Data Transformation helpers
"""
def _to_categorical(y: list, num_classes) -> np.ndarray:
    '''
    Converts a list of integers to one-hot encoding
    '''
    return F.one_hot(y, num_classes=num_classes).numpy()  # Convert to numpy

def _get_resized_img(char_image: np.ndarray) -> np.ndarray:
    return cv2.resize(char_image, (IMG_WIDTH, IMG_HEIGHT))

def _get_transformed_data_helper(folder_path, is_train, segmentation_function):
    """
    Segments input X based on segmentation_function (which takes in a list of captchas and returns a list of list of character images)
    and returns (X,y) as tensors.
    """
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    X, y = [], []
    desc = f"Preparing {'Train' if is_train else 'Test'} Data"
    for filename in tqdm(filenames, desc=desc):
        img_path = os.path.join(folder_path, filename)
        filename_without_suffix = os.path.splitext(filename)[0]
        correct_characters = filename_without_suffix.split('-')[0]

        image = cv2.imread(img_path)
        preprocessed_image = preprocess_image_v1(image)
        char_imgs = segmentation_function([preprocessed_image])[0]

        # Skip if segmentation failed
        if len(char_imgs) != len(correct_characters):
            continue

        # Add every input character and its label to the (X,y) dataset
        X.extend([_get_resized_img(img) for img in char_imgs])
        y.extend([CHARACTERS.index(char_label) for char_label in correct_characters])
    
    # Transform (X,y) to tensors for PyTorch
    X = np.array(X, dtype=np.float32) / 255.0 # Normalize
    X = np.expand_dims(X, axis=1) # Reshape to (N, 1, 40, 30) for CNN
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

def get_transformed_data_for_captcha_evaluation_helper(folder_path, segmentation_function) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, int]]:
    """
    Rationale is to allow for evaluation of CAPTCHA performance now instead of just individual characters, which includes 
    number of failed segmentations.
    Now returns (X,y) with an additional layer of nesting for grouping character images and their labels for each CAPTCHA.
    
    Also segments based on segmentation_function (which takes in a list of captchas and returns a list of list of character images)
    """
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    X_test_captcha, y_test_captcha = [], []
    num_failed_segmentations = 0
    num_failed_chars = 0
    
    desc = f"Preparing Test Data for CAPTCHA Evaluation"
    for filename in tqdm(filenames, desc=desc):
        img_path = os.path.join(folder_path, filename)
        filename_without_suffix = os.path.splitext(filename)[0]
        correct_characters = filename_without_suffix.split('-')[0]

        image = cv2.imread(img_path)
        preprocessed_image = preprocess_image_v1(image)
        char_imgs = segmentation_function([preprocessed_image])[0]

        # Keep track of number of failed CAPTCHA segmentations
        if len(char_imgs) != len(correct_characters):
            num_failed_segmentations += 1
            num_failed_chars += len(correct_characters)
            continue

        # Treat each CAPTCHA as a (X,y) dataset of character images and labels
        X = [_get_resized_img(img) for img in char_imgs]
        y = [CHARACTERS.index(char_label) for char_label in correct_characters]

        # Transform (X,y) for each CAPTCHA to tensors for PyTorch, and add them to X_test_captcha, y_test_captcha
        X = np.array(X, dtype=np.float32) / 255.0 # Normalize
        X = np.expand_dims(X, axis=1) # Reshape to (N, 1, 40, 30) for CNN
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        X_test_captcha.append(X)
        y_test_captcha.append(y)

    return X_test_captcha, y_test_captcha, (num_failed_segmentations, num_failed_chars)


"""
[Public] Data Transformation methods
"""
def get_transformed_data(folder_path, is_train) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, StandardScaler]:
    """
    Denoise and tokenize captcha image files in a folder into individual characters using vertical projection.
    Ignores image files where segmentation has failed (num_of_segmented_char =/= actual_num_of_char).

    Parameters
       folder_path (str): Path to the folder containing CAPTCHA images. Image files in the folder should be of 
       format "captchachars-0" （e.g "abc123-0")

    Returns:
        tuple:
            - X_imgs：Numpy array of Image of char with type numpy array (40 x 30)
            - y: Numpy array of one-hot encoded label of the corresponding image
    """
    all_images = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    X_imgs = []  # Character images
    X_features = []  # Engineered features for each character
    y = []  # Labels
    desc = f"Preparing {'Train' if is_train else 'Test'} Data"
    for filename in tqdm(all_images, desc="Preparing Data"):
        img_path = os.path.join(folder_path, filename)
        filename_without_suffix = os.path.splitext(filename)[0]
        correct_characters = filename_without_suffix.split('-')[0]

        image = cv2.imread(img_path)
        preprocessed_image = preprocess_image_v1(image)
        character_boundaries = segment_by_projection_v2(preprocessed_image, PROJECTION_THRESHOLD)

        # Skip if segmentation failed or number of segments doesn't match expected characters
        if len(character_boundaries) != len(correct_characters):
            continue

        # Add every input character and its label to the (X,y) dataset, and also extract features 
        # for each character
        for i, (start, end) in enumerate(character_boundaries):
            char_image = preprocessed_image[:, start:end]
            char_label = correct_characters[i]

            # Skip if character is not in our expected set
            if char_label not in CHARACTERS:
                continue

            # Extract character features
            features = extract_features(char_image)

            # Add to dataset
            X_imgs.append(_get_resized_img(char_image))
            X_features.append(features)
            y.append(CHARACTERS.index(char_label))


    # Transform X_imgs
    X_imgs = np.array(X_imgs)
    X_imgs = np.expand_dims(X_imgs, 1) # Reshape for CNN
    X_imgs = X_imgs / 255.0  # Normalize
    X_imgs = X_imgs.astype(np.float32)  # Convert to float for pytorch

    # Transform y
    y = torch.tensor(y, dtype=torch.long)
    
    # Transform X_features and standardise
    features_df = pd.DataFrame(X_features)
    feature_names = list(features_df.columns)
    feature_values = features_df.values

    scaler = StandardScaler()
    X_feature_vectors = scaler.fit_transform(feature_values).astype(np.float32)
    
    return X_imgs, y, feature_names, X_feature_vectors, scaler


def get_transformed_data_v2(folder_path, is_train) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_transformed_data_helper(folder_path, is_train, segment_by_projection_v1)


def get_transformed_data_v2_with_padding(folder_path, is_train) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_transformed_data_helper(folder_path, is_train, segment_by_projection_v1_with_padding)


def get_transformed_data_for_captcha_evaluation(folder_path) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, int]]:
    return get_transformed_data_for_captcha_evaluation_helper(folder_path, segment_by_projection_v1)


def get_transformed_data_for_captcha_evaluation_with_padding(folder_path) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, int]]:
    return get_transformed_data_for_captcha_evaluation_helper(folder_path, segment_by_projection_v1_with_padding)