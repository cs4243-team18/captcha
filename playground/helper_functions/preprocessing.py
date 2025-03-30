import cv2
import numpy as np


"""
Preprocessing methods for Phase 1 of the pipeline.

Note: [Private] methods refer to internal helper functions that stay within this package, while [Public] methods 
are the ones meant to be exported for use by other packages.
"""


"""
[Private] Denoising methods
"""
def replace_black_with_median(image: np.ndarray, kernel_size=5) -> np.ndarray:
    """
    Replaces all black pixels in an image with the median of their local neighborhood.
    
    Parameters:
        image: Input image (grayscale or color)
        kernel_size: Size of the patch used for computing the median
    Returns:
        denoised_image: Image with black pixels replaced by median values
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


def remove_salt_and_pepper_noise_by_morphology(image: np.ndarray, kernel_size = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


"""
[Private] Binarization methods
"""
def binarization_max_contrast(image: np.ndarray) -> np.ndarray:
    """
    To prevent missing out on text characters which are too bright or light-colored, compare
    grayscale, HSV Value (V) channel, and LAB Lightness (L) channel to pick the best channel
    for binarization.
    """
    # Get grayscale, HSV V, and LAB L channels
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
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


"""
[Public] Preprocessing methods (using a mix and match of various denoising and binarization methods)
"""
def preprocess_image_v1(image: np.ndarray) -> np.ndarray:
    """
    Applies black pixel median filtering, salt and pepper removal, then max contrast binarization.
    """
    denoised_image = replace_black_with_median(image.copy())
    denoised_image = remove_salt_and_pepper_noise_by_morphology(denoised_image)
    return binarization_max_contrast(denoised_image)


def preprocess_image_v2(image: np.ndarray) -> np.ndarray:
    """
    Applies black pixel median filtering, salt and pepper removal, then <some other binarization method>.
    """
    pass
    
    
