from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import json
from helper_functions.data_transformation import CHARACTERS

"""
[Public] Functions for evaluating the character and CAPTCHA performance of trained PyTorch models.

NOTE: Char accuracy A at the character level skips failed segmentations, while char accuracy B at the CAPTCHA
level does not, hence B < A. Example: If 'abcde' failed to segment, then A will skip that, but B will treat it
as 0/5 chars identified correctly.
"""


def evaluate_character_performance(
    trained_model: nn.Module, 
    X_test: torch.Tensor, 
    y_test: torch.Tensor,
):
    """
    Uses the trained CNN model to predict and evaluate its character recognition performance.
    """
    NUM_CLASSES = len(CHARACTERS)
    trained_model.eval()

    # Get predictions and convert tensors to numpy arrays for calculations
    with torch.no_grad():
        predictions = trained_model(X_test)
    y_pred = torch.argmax(predictions, axis=1).cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Calculate macro character performance metrics
    accuracy_macro = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', labels=np.arange(NUM_CLASSES), zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', labels=np.arange(NUM_CLASSES), zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', labels=np.arange(NUM_CLASSES), zero_division=0)

    # Structure the metrics and print them out
    character_performance = {
        'accuracy': float(accuracy_macro),
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1_score': float(f1_macro)
    }

    # Round everything to 3dp
    for key in character_performance.keys():
        character_performance[key] = round(character_performance[key], 3)

    print(f"Character level performance (skipping SF): {json.dumps(character_performance, indent=2)}\n")


def evaluate_individual_character_performance(    
    trained_model: nn.Module, 
    X_test: torch.Tensor, 
    y_test: torch.Tensor,
    top_n_confusions: int = 3
):
    """
    Similar as above, but for per-character performance metrics and confusion analysis.
    """
    NUM_CLASSES = len(CHARACTERS)
    trained_model.eval()

    # Get predictions and convert tensors to numpy arrays for calculations
    with torch.no_grad():
        predictions = trained_model(X_test)
    y_pred = torch.argmax(predictions, axis=1).cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Calculate per-character performance metrics
    # Note: The 'None' parameter for average returns scores for each class, instead of avg for all like in macro metrics
    precision_per_char = precision_score(y_test, y_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    recall_per_char = recall_score(y_test, y_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    f1_per_char = f1_score(y_test, y_pred, average=None, labels=np.arange(NUM_CLASSES), zero_division=0)
    per_char_accuracy = [] # For per-character accuracy, need to calculate manually
    for char_idx in range(NUM_CLASSES):
        char_mask = (y_test == char_idx)
        if np.sum(char_mask) > 0: 
            correct_predictions = (y_pred[char_mask] == char_idx)
            char_accuracy = np.mean(correct_predictions)
            per_char_accuracy.append(char_accuracy)
        else:
            per_char_accuracy.append(0) 
    
    # Visualise confusion matrix heatmap
    print(f"Confusion matrix for individual characters (skipping SF):")
    confusion_mat = confusion_matrix(y_test, y_pred, labels=np.arange(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        annot_kws={"size": 8},
        fmt="d",
        cmap="Blues",
        xticklabels=list(CHARACTERS),
        yticklabels=list(CHARACTERS),
        norm = mcolors.BoundaryNorm([0, 1, 5, 10, 20, 50, 100, 200, 300, confusion_mat.max()], ncolors=256)
    )
    plt.yticks(rotation=0), plt.xlabel("Predicted Label"), plt.ylabel("True Label"), plt.title("Confusion Matrix Heatmap")
    plt.show()
    
    # Create per-character performance dictionary with confusion analysis
    per_character_metrics = {}
    confusion_analysis = {}
    
    for i in range(NUM_CLASSES):
        char = CHARACTERS[i]
        # Basic metrics
        per_character_metrics[char] = {
            'accuracy': float(per_char_accuracy[i]),
            'precision': float(precision_per_char[i]),
            'recall': float(recall_per_char[i]),
            'f1_score': float(f1_per_char[i])
        }
        
        # Confusion analysis
        true_cases = np.sum(y_test == i)
        if true_cases > 0:
            confusions = confusion_mat[i, :] # Extract the row from confusion matrix (what this character was confused as)
            confusions[i] = 0 # Set the diagonal (correct predictions) to zero to focus on errors
            confusion_percentages = confusions / true_cases if true_cases > 0 else np.zeros_like(confusions) # Normalise
            top_confusions_indices = np.argsort(-confusion_percentages)[:top_n_confusions] # Get top N confusions
            top_confusions_list = []
            for j in top_confusions_indices:
                if confusion_percentages[j] == 0:
                    continue
                confused_char = CHARACTERS[j]
                confusion_percent = float(confusion_percentages[j] * 100)
                confusion_count = int(confusions[j])
                top_confusions_list.append({
                    'confused_as': confused_char,
                    'percentage': confusion_percent,
                    'count': confusion_count
                })
            confusion_analysis[char] = {
                'total_samples': int(true_cases),
                'top_confusions': top_confusions_list
            }
        else:
            confusion_analysis[char] = {
                'total_samples': 0,
                'top_confusions': []
            }

    # Sort character metrics and confusion analysis by accuracy (lowest first)
    sorted_metrics = dict(sorted(
        per_character_metrics.items(), 
        key=lambda pair: pair[1]['accuracy']
    ))
    sorted_confusion = {}
    for char in sorted_metrics.keys():
        sorted_confusion[char] = confusion_analysis[char]
    
    # Visualise the metrics and top confusions in a table plotted by plt.table
    print(f"Individual character performance and top confusions (skipping SF):")
    data = {
        "Character": [char for char in sorted_metrics.keys()],
        "Top 3 Misclassifications": [
            (' '*6).join([f"{info['confused_as']} ({info['count']})" for info in metrics["top_confusions"]]) 
            for metrics in sorted_confusion.values()
        ],
        "Total samples": [metrics["total_samples"] for metrics in sorted_confusion.values()],
        "Accuracy": [round(metrics["accuracy"], 2) for metrics in sorted_metrics.values()],
        "Precision": [round(metrics["precision"], 2) for metrics in sorted_metrics.values()],
        "Recall": [round(metrics["recall"], 2) for metrics in sorted_metrics.values()],
        "F1-score": [round(metrics["f1_score"], 2) for metrics in sorted_metrics.values()],
    }
    df = pd.DataFrame(data)

    _, axis = plt.subplots(figsize=(10, 10))
    # Hide the axis
    axis.xaxis.set_visible(False), axis.yaxis.set_visible(False), axis.set_frame_on(False)
    # Create the table
    table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    # Adjust layout
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(df.columns))])
    table.scale(1.2,1.2)
    
    plt.show()
        

def evaluate_captcha_performance(
    trained_model: nn.Module, 
    X_test_captcha: list[torch.Tensor], 
    y_test_captcha: list[torch.Tensor], 
    failed_segmentation_info: tuple[int, int]
):
    """
    Similar to above, but for CAPTCHA level performance and includes failed segmentations.
    """
    trained_model.eval()

    num_correct_captchas = 0
    num_correct_chars = 0
    total_chars = 0

    for captcha_x, captcha_y in zip(X_test_captcha, y_test_captcha):
        y_pred = torch.argmax(trained_model(captcha_x), axis=1).cpu().numpy()
        y_true = captcha_y.cpu().numpy()
        num_correct_chars += np.sum(y_pred == y_true)
        total_chars += len(y_true)
        num_correct_captchas +=  np.all(y_pred == y_true)

    num_failed_segmentations, num_failed_chars = failed_segmentation_info
    total_captchas = len(X_test_captcha) + num_failed_segmentations
    total_captchas_skip_SF = len(X_test_captcha)
    total_chars += num_failed_chars

    captcha_performance = {
        'segmentation_accuracy': 1 - (num_failed_segmentations / total_captchas),
        'captcha_accuracy (skipping SF)': num_correct_captchas / total_captchas_skip_SF,
        'captcha_accuracy (including SF)': num_correct_captchas / total_captchas,
        'character_accuracy (including SF)': num_correct_chars / total_chars,
    }
    
    # Round everything to 3dp
    for key in captcha_performance.keys():
        captcha_performance[key] = round(captcha_performance[key], 3)

    print(f"Captcha level performance: {json.dumps(captcha_performance, indent=2)}\n")
    

def visualize_cnn_features(model, X, cols=16):
    feature_maps = model.get_feature_maps(X)
    for layer_name, feat_map in feature_maps.items():
        batch_size = feat_map.shape[0]
        num_filters = feat_map.shape[1]

        for b in range(batch_size):
            # Show original image first
            if X is not None:
                plt.figure(figsize=(1, 1))
                img = X[b]
                if img.shape[0] == 1:  # grayscale
                    plt.imshow(img.squeeze().detach().cpu(), cmap='gray')
                else:
                    plt.imshow(img.permute(1, 2, 0).detach().cpu())
                plt.title(f"Original Image {b}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            # Then show the feature maps
            rows = (num_filters + cols - 1) // cols
            plt.figure(figsize=(int(cols * 0.7), rows))
            for i in range(num_filters):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(feat_map[b, i].detach().cpu(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"{layer_name} | Image {b}")
            plt.tight_layout()
            plt.show()