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

def segment_captcha_with_projection(image, projection_threshold):
    vertical_projection = np.sum(image, axis=0)
    max_projection = vertical_projection.max()
    if max_projection != 0:
        vertical_projection = vertical_projection / max_projection

    projection_binary = vertical_projection > projection_threshold

    character_boundaries = []
    start_idx = None
    for i in range(1, len(projection_binary)):
        if projection_binary[i] != projection_binary[i - 1]:
            if projection_binary[i] == 1:
                start_idx = i
            elif projection_binary[i] == 0 and start_idx is not None:
                end_idx = i
                character_boundaries.append((start_idx, end_idx))
                start_idx = None
    
    # Handle the case where the last character extends to the end
    if start_idx is not None:
        character_boundaries.append((start_idx, len(projection_binary)))
    
    return character_boundaries, vertical_projection, projection_binary