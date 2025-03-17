import os
import cv2
import random
from matplotlib import pyplot as plt

"""
Image loading functions
"""

def get_png_filepaths(train_imgs_dir):
    png_files = []
    
    # Now use this absolute path
    subdirs = [os.path.join(train_imgs_dir, d) for d in os.listdir(train_imgs_dir)
               if os.path.isdir(os.path.join(train_imgs_dir, d))]
    for subdir in subdirs:
        for filename in os.listdir(subdir):  # List files inside the subdirectory
            if filename.lower().endswith(".png"):
                png_files.append(os.path.join(subdir, filename))  # Store full path

    return png_files


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