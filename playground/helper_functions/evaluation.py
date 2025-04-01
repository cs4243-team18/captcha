import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


"""
[Public] Functions for evaluating the character and CAPTCHA performance of trained PyTorch models.

NOTE: Char accuracy A at the character level skips failed segmentations, while char accuracy B at the CAPTCHA
level does not, hence B < A. Example: If 'abcde' failed to segment, then A will skip that, but B will treat it
as 0/5 chars identified correctly.
"""


def evaluate_character_performance(
    trained_model: nn.Module, 
    X_test: torch.Tensor, 
    y_test: torch.Tensor
) -> dict:
    """
    Uses the trained CNN model to predict and evaluate its character recognition performance.
    
    Parameters
    X_test: tensor of m testing character images, where each grayscale character image is a normalised matrix 
        of shape (IMG_HEIGHT, IMG_WIDTH)

    Returns
    character_performance: dict of character level perfor
    """
    trained_model.eval()

    # Convert tensors to numpy arrays for calculations
    y_pred = torch.argmax(trained_model(X_test), axis=1).cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Calculate macro performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', labels=np.arange(36), zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', labels=np.arange(36), zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', labels=np.arange(36), zero_division=0)

    # Calculate micro performance metrics (i.e. for each of the 36 characters)


    character_performance = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return character_performance


def evaluate_captcha_performance(
    trained_model: nn.Module, 
    X_test_captcha: list[torch.Tensor], 
    y_test_captcha: list[torch.Tensor], 
    num_failed_segmentations: int
) -> dict:
    """
    Similar to above, but for CAPTCHA level performance instead.
    """
    trained_model.eval()
    
    num_correct_captchas = 0
    total_captchas = len(X_test_captcha) + num_failed_segmentations

    segmentation_accuracy = len(X_test_captcha) / total_captchas

    num_correct_chars = 0
    total_chars = 0

    for captcha_x, captcha_y in zip(X_test_captcha, y_test_captcha):
        y_pred = torch.argmax(trained_model(captcha_x), axis=1).cpu().numpy()
        y_true = captcha_y.cpu().numpy()
        num_correct_chars += np.sum(y_pred == y_true)
        total_chars += len(y_true)
        num_correct_captchas +=  np.all(y_pred == y_true)

    captcha_performance = {
        'captcha_accuracy': num_correct_captchas / total_captchas,
        'character_accuracy': num_correct_chars / total_chars,
        'segmentation_accuracy': segmentation_accuracy,
    }
    return captcha_performance
