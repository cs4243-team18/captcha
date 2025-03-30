import numpy as np

from typing import List

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