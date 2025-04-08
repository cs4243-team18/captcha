import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

"""
Image loading functions
"""

def get_png_filepaths(train_imgs_dir):
    return [f"{train_imgs_dir}/{file}" for file in os.listdir(train_imgs_dir) if file.endswith('.png')]

def get_random_train_img(train_imgs_dir):
    return cv2.imread(random.choice(get_png_filepaths(train_imgs_dir)))

def get_n_random_train_imgs(train_imgs_dir, n):
    filepaths = random.choices(get_png_filepaths(train_imgs_dir), k=n)
    return [cv2.imread(filepath) for filepath in filepaths]

"""
Visualisation functions
"""
def imshow(img, title, subplot=None):
    if subplot:
        plt.subplot(subplot)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title(title)

def imshow_bulk(img, title, plot):
    plot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plot.set_title(title)

def get_combined_char_imgs(char_imgs):
    # Add space in between char imgs, then hstack them together
    char_imgs_with_spacing = []
    for char_img in char_imgs:
        char_imgs_with_spacing.append(char_img)
        spacing = np.ones_like(char_img) * 255
        char_imgs_with_spacing.append(spacing)
    return np.hstack(char_imgs_with_spacing)

import numpy as np

def get_combined_char_imgs_v2(char_imgs):
    # Find the max height among all char images
    max_height = max(img.shape[0] for img in char_imgs)
    
    char_imgs_with_spacing = []
    
    for char_img in char_imgs:
        h, w = char_img.shape
        
        # Pad each char_img to the same height (max_height)
        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        padded_char_img = np.pad(char_img, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=255)  # Padding with white
        
        # Add the padded char_img to the list
        char_imgs_with_spacing.append(padded_char_img)
        
        # Add the spacing (white area) after each image
        spacing = np.ones_like(padded_char_img) * 255  # Same height as the image
        char_imgs_with_spacing.append(spacing)
    
    # Remove the last spacing to avoid extra space at the end
    if char_imgs_with_spacing:
        char_imgs_with_spacing = char_imgs_with_spacing[:-1]

    # Stack all the images horizontally
    return np.hstack(char_imgs_with_spacing)
