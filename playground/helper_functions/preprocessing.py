import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import string
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler

"""
Denoising methods
"""

def replace_black_with_median(image, kernel_size=5):
    """
    Replaces all black pixels in an image with the median of their local neighborhood.
    
    :param image: Input image (grayscale or color)
    :param kernel_size: Size of the patch used for computing the median
    :return: Image with black pixels replaced by median values
    """
    is_color = len(image.shape) == 3
    
    # Define black pixel condition
    if is_color:
        black_pixels = np.all(image == [0, 0, 0], axis=-1)  # Mask for black pixels
    else:
        black_pixels = (image == 0)  # Grayscale mask

    # Compute median blur (preserves edges better than Gaussian blur)
    median_filtered = cv2.medianBlur(image, kernel_size)
    
    # Replace black pixels with median values
    image[black_pixels] = median_filtered[black_pixels]

    return image

"""
Salt and pepper filter
"""
def remove_salt_and_pepper_noise(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def remove_salt_and_pepper_noise(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


"""
Binarization methods
"""
def binarization_max_contrast(img):
    """
    To prevent missing out on text characters which are too bright or light-colored, compare
    grayscale, HSV Value (V) channel, and LAB Lightness (L) channel to pick the best channel
    for binarization.
    """
    # Get grayscale, HSV V, and LAB L channels
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Pick the best channel dynamically (whichever has the highest variance)
    choices = [grayscale, v_channel, l_channel]
    best_channel = max(choices, key=lambda x: np.var(x))  # Higher variance = better contrast

    # Apply Adaptive Thresholding for dynamic text binarization
    binarized = cv2.adaptiveThreshold(
        best_channel, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    return binarized


