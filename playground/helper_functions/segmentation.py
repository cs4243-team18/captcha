import cv2
import numpy as np
import os
from typing import List
from tqdm import tqdm

from helper_functions.preprocessing import preprocess_image_v1
IMG_HEIGHT = 40
IMG_WIDTH = 30
"""
Segmentation methods for Phase 2 of the pipeline
"""

"""
[Public] Segmentation methods by projection
"""
def segment_by_projection_v1(binarised_imgs: list[np.ndarray]) -> List[list[np.ndarray]]:
    """
    Given a list of binarised images, return a list of segmented characters for each binarised image
    """
    # Compute vertical projection (sum of pixel values along columns)
    vertical_projs = [np.sum(binary, axis=0) for binary in binarised_imgs]

    # Identify gap points (valleys where projection value is low), then derive horizontal segments of characters
    thresholds = [np.max(vertical_proj) * 0.05 for vertical_proj in vertical_projs] # Adjust threshold as needed
    all_x_gap_points = [np.where(vertical_proj < threshold)[0] for vertical_proj, threshold in zip(vertical_projs, thresholds)]
    all_x_segments = []
    for x_gap_points in all_x_gap_points:
        x_segments = []
        prev = 0
        for point in x_gap_points:
            if point - prev > 2:  # Adjust spacing threshold
                x_segments.append((prev, point))
            prev = point
        all_x_segments.append(x_segments)

    # Compute horizontal projection (sum of pixel values along rows) to trim the vertical axis
    horizontal_projs = [np.sum(binary, axis=1) for binary in binarised_imgs]

    # Find min_y and max_y for each image
    all_y_bounds = []
    for horizontal_proj in horizontal_projs:
        nonzero_indices = np.where(horizontal_proj > 0)[0]  # Find rows containing characters
        if nonzero_indices.size > 0:
            min_y, max_y = max(0,nonzero_indices[0]-10), nonzero_indices[-1]+10
        else:
            min_y, max_y = 0, horizontal_proj.shape[0] - 1  # If the image is empty, keep full height
        all_y_bounds.append((min_y, max_y))

    # Extract individual characters using bounding boxes
    all_char_images_vert_proj = []
    for x_segments, binary, y_bounds in zip(all_x_segments, binarised_imgs, all_y_bounds):
        char_images = []
        for start,end in x_segments:
            char_images.append(binary[y_bounds[0]:y_bounds[1]+1, start:end+1])
        # Remove any noise (> 95% zeros)
        char_images = [char for char in char_images if not np.count_nonzero(char == 0) / char.size > 0.95]
        all_char_images_vert_proj.append(char_images)
    return all_char_images_vert_proj

def proportionate_resize(char_img, dimension):
    """
    Proportionately resize character images, and adds padding 
    """
    height, width = char_img.shape
    y_scale = float(height)/dimension[0]
    x_scale = float(width)/dimension[1]
    if y_scale > x_scale:
        # new_width need to be minimum of 1 to consider rare cases where noise is considered as a char_img
        new_width = max(1, int(width / y_scale))
        new_img = cv2.resize(char_img, (new_width, dimension[0]))
        padding = (dimension[1] - new_width) // 2
        new_img = np.pad(new_img, ((0,0),(padding,padding)) , mode='constant', constant_values=0)
    elif x_scale> y_scale:
        # new_width need to be minimum of 1 to consider rare cases where noise is considered as a char_img
        new_height = max(1, int(height / x_scale))
        new_img = cv2.resize(char_img, (dimension[1], new_height))
        padding = (dimension[0] - new_height) // 2
        new_img = np.pad(new_img, ((padding,padding), (0,0)), mode='constant', constant_values=0)
    else:
        new_img = cv2.resize(char_img, (dimension[1], dimension[0]))
    return new_img
    
def segment_by_projection_v1_proportionate(binarised_imgs: list[np.ndarray]) -> List[list[np.ndarray]]:
    """
    Given a list of binarised images, return a list of segmented characters for each binarised image
    """
    
    # Compute vertical projection (sum of pixel values along columns)
    vertical_projs = [np.sum(binary, axis=0) for binary in binarised_imgs]

    # Identify gap points (valleys where projection value is low), then derive horizontal segments of characters
    thresholds = [np.max(vertical_proj) * 0.05 for vertical_proj in vertical_projs] # Adjust threshold as needed
    all_x_gap_points = [np.where(vertical_proj < threshold)[0] for vertical_proj, threshold in zip(vertical_projs, thresholds)]
    all_x_segments = []
    for x_gap_points in all_x_gap_points:
        x_segments = []
        prev = 0
        for point in x_gap_points:
            if point - prev > 2:  # Adjust spacing threshold
                x_segments.append((prev, point))
            prev = point
        all_x_segments.append(x_segments)
    
    # Find min_y and max_y for each image
    horizontal_projs = [np.sum(binary, axis=1) for binary in binarised_imgs]
    all_y_bounds = []
    for horizontal_proj in horizontal_projs:
        nonzero_indices = np.where(horizontal_proj > 0)[0]  # Find rows containing characters
        if nonzero_indices.size > 0:
            min_y, max_y = max(0,nonzero_indices[0]-10), nonzero_indices[-1]+10
        else:
            min_y, max_y = 0, horizontal_proj.shape[0] - 1  # If the image is empty, keep full height
        all_y_bounds.append((min_y, max_y))

    # Extract individual characters using bounding boxes
    all_char_images_vert_proj = []
    for x_segments, binary, y_bounds in zip(all_x_segments, binarised_imgs, all_y_bounds):
        char_images = []
        for start,end in x_segments:
            char_images.append(binary[y_bounds[0]:y_bounds[1]+1, start:end+1])
        # Remove any noise (> 95% zeros)
        char_images = [char for char in char_images if not np.count_nonzero(char == 0) / char.size > 0.95]
        
        # Scaled proportionally char images
        proportionate_char_images = []
        for image in char_images:
            horizontal_proj = np.sum(image, axis=1)
            nonzero_indices = np.where(horizontal_proj > 0)[0]
            min_y, max_y = max(0,nonzero_indices[0]), nonzero_indices[-1]
            proportionate_char_images.append(proportionate_resize(image[min_y:max_y+1, : ],(IMG_HEIGHT, IMG_WIDTH)))
        
        all_char_images_vert_proj.append(proportionate_char_images)
    return all_char_images_vert_proj

def segment_by_projection_v1_with_padding(binarised_imgs: list[np.ndarray]) -> List[list[np.ndarray]]:
    """
    Given a list of binarised images, return a list of segmented characters for each binarised image
    """
    # Compute vertical projection (sum of pixel values along columns)
    vertical_projs = [np.sum(binary, axis=0) for binary in binarised_imgs]

    # Identify gap points (valleys where projection value is low), then derive horizontal segments of characters
    thresholds = [np.max(vertical_proj) * 0.05 for vertical_proj in vertical_projs] # Adjust threshold as needed
    all_x_gap_points = [np.where(vertical_proj < threshold)[0] for vertical_proj, threshold in zip(vertical_projs, thresholds)]
    all_x_segments = []
    for x_gap_points in all_x_gap_points:
        x_segments = []
        prev = 0
        for point in x_gap_points:
            if point - prev > 2:  # Adjust spacing threshold
                x_segments.append((prev, point))
            prev = point
        all_x_segments.append(x_segments)

    # Compute horizontal projection (sum of pixel values along rows) to trim the vertical axis
    horizontal_projs = [np.sum(binary, axis=1) for binary in binarised_imgs]

    # Find min_y and max_y for each image
    all_y_bounds = []
    for horizontal_proj in horizontal_projs:
        nonzero_indices = np.where(horizontal_proj > 0)[0]  # Find rows containing characters
        if nonzero_indices.size > 0:
            min_y, max_y = max(0,nonzero_indices[0]-3), nonzero_indices[-1]+3
        else:
            min_y, max_y = 0, horizontal_proj.shape[0] - 1  # If the image is empty, keep full height
        all_y_bounds.append((min_y, max_y))

    # Extract individual characters using bounding boxes
    all_char_images_vert_proj = []
    for x_segments, binary, y_bounds in zip(all_x_segments, binarised_imgs, all_y_bounds):
        char_images = []
        for start,end in x_segments:
            char_images.append(binary[y_bounds[0]:y_bounds[1]+1, start:end+1])
        # Remove any noise (> 94% zeros)
        char_images = [char for char in char_images if not np.count_nonzero(char == 0) / char.size > 0.94]
        # Add padding to each character image
        min_width = 0.5*(y_bounds[1] - y_bounds[0])
        char_images = [
            np.pad(char, ((0, 0), ((pad := max(0, int(min_width) - char.shape[1])) // 2, pad - (pad // 2))), mode='constant', constant_values=0)
            if char.shape[1] < int(min_width) else char
            for char in char_images
        ]
        all_char_images_vert_proj.append(char_images)
    return all_char_images_vert_proj


def segment_by_projection_v2(image: np.ndarray, projection_threshold=0.1) -> list[tuple[int, int]]:
    vertical_projection = np.sum(image, axis=0)
    max_projection = vertical_projection.max()
    if max_projection != 0:
        vertical_projection = vertical_projection / max_projection

    projection_binary = vertical_projection > projection_threshold
    x_len = len(projection_binary)

    character_boundaries = []
    start_idx = None
    for i in range(1, x_len):
        if projection_binary[i] == projection_binary[i-1]:
            continue
        if projection_binary[i] == 1:
            start_idx = i
        elif projection_binary[i] == 0 and start_idx is not None:
            end_idx = i
            character_boundaries.append((start_idx, end_idx))
            start_idx = None
    
    # Handle the case where the last character extends to the end
    if start_idx is not None:
        character_boundaries.append((start_idx, x_len))
    
    return character_boundaries


def segment_by_projection_v3(binarised_imgs: list[np.ndarray]) -> List[list[np.ndarray]]:
    """
    Given a list of binarised images, return a list of segmented characters for each binarised image
    """
    # Compute vertical projection (sum of pixel values along columns)
    vertical_projs = [np.sum(binary, axis=0) for binary in binarised_imgs]

    # Identify gap points (valleys where projection value is low), then derive horizontal segments of characters
    thresholds = [np.max(vertical_proj) * 0.05 for vertical_proj in vertical_projs] # Adjust threshold as needed
    all_x_gap_points = [np.where(vertical_proj < threshold)[0] for vertical_proj, threshold in zip(vertical_projs, thresholds)]
    all_x_segments = []
    for x_gap_points in all_x_gap_points:
        x_segments = []
        prev = 0
        for point in x_gap_points:
            if point - prev > 2:  # Adjust spacing threshold
                x_segments.append((prev, point))
            prev = point
        all_x_segments.append(x_segments)

    # Find min_y and max_y for each image
    horizontal_projs = [np.sum(binary, axis=1) for binary in binarised_imgs]
    all_y_bounds = []
    for horizontal_proj in horizontal_projs:
        nonzero_indices = np.where(horizontal_proj > 0)[0]  # Find rows containing characters
        if nonzero_indices.size > 0:
            min_y, max_y = max(0,nonzero_indices[0]-10), nonzero_indices[-1]+10
        else:
            min_y, max_y = 0, horizontal_proj.shape[0] - 1  # If the image is empty, keep full height
        all_y_bounds.append((min_y, max_y))

    # Extract individual characters using bounding boxes
    all_char_images_vert_proj = []
    for x_segments, binary, y_bounds in zip(all_x_segments, binarised_imgs, all_y_bounds):
        char_images = []
        for start,end in x_segments:
            img = binary[y_bounds[0]:y_bounds[1]+1, start:end+1]
            # Remove any noise (> 95% zeros)
            if np.count_nonzero(img == 0) / img.size > 0.95:
                continue

            # Compute horizontal projection (sum of pixel values along rows) to trim the vertical axis
            horizontal_proj = np.sum(img, axis=1)
            nonzero_indices = np.where(horizontal_proj > 0)[0]
            if nonzero_indices.size == 0: continue
            y_bottom, y_top = nonzero_indices[0], nonzero_indices[-1]
            trimmed = img[y_bottom-3:y_top+3, :]  
            # Pad if too thick or thin
            min_height = 1.3 * (end - start)
            min_width = 0.75 * (y_top - y_bottom)
            if trimmed.shape[0] < min_height:
                pad = int(min_height) - trimmed.shape[0]
                pad_top = pad // 2
                pad_bottom = pad - pad_top
                trimmed = np.pad(trimmed, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
            if trimmed.shape[1] < min_width:
                pad = int(min_width) - trimmed.shape[1]
                pad_left = pad // 2
                pad_right = pad - pad_left
                trimmed = np.pad(trimmed, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

            char_images.append(trimmed)

        all_char_images_vert_proj.append(char_images)
    return all_char_images_vert_proj

"""
[Public] Segmentation evaluation methods
"""

def _evaluate_segmentation_accuracy(folder_path: str, segmentation_function, title: str):
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    num_failed_segmentations = 0

    for filename in tqdm(filenames, desc=f"Evaluating Segmentation Accuracy for Vertical Projection {title}"):
        img_path = os.path.join(folder_path, filename)
        filename_without_suffix = os.path.splitext(filename)[0]
        correct_characters = filename_without_suffix.split('-')[0]

        image = cv2.imread(img_path)
        preprocessed_image = preprocess_image_v1(image)
        char_images = segmentation_function([preprocessed_image])[0]

        if len(char_images) != len(correct_characters):
            num_failed_segmentations += 1
    total_files = len(filenames)
    total_successful_segmentations = total_files - num_failed_segmentations
    print(f"{title} accuracy: {total_successful_segmentations*100 / total_files} % ({total_successful_segmentations} out of {total_files})")


def evaluate_segmentation_accuracy_v1(folder_path):
    _evaluate_segmentation_accuracy(folder_path, segment_by_projection_v1, "V1")

def evaluate_segmentation_accuracy_v1_with_padding(folder_path):
    _evaluate_segmentation_accuracy(folder_path, segment_by_projection_v1_with_padding, "V1 with Padding")

def evaluate_segmentation_accuracy_v2(folder_path):
    _evaluate_segmentation_accuracy(folder_path, segment_by_projection_v2, "V2")

def evaluate_segmentation_accuracy_v3(folder_path):
    _evaluate_segmentation_accuracy(folder_path, segment_by_projection_v3, "V3")