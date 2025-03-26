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


"""
Evaluation functions
"""
