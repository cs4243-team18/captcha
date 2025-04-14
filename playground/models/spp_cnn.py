import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import TypedDict, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from helper_functions.data_transformation import CHARACTERS

class CNNParams(TypedDict):
    num_classes: int
    input_shape: Tuple[int, int, int]  # (channels, height, width)
    learning_rate: float
    num_epochs: int
    batch_size: int
    spp_levels: List[int]  # Pyramid levels for SPP


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels):
        """
        Spatial Pyramid Pooling layer that converts feature maps of any size into fixed-length outputs.
        Args:
            levels: List of output bin sizes (e.g., [1, 2, 4] for 1x1, 2x2, and 4x4 bins)
        """
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels
        
    def forward(self, x):
        batch_size = x.size(0)
        features = []
        
        for level in self.levels:
            # Use adaptive pooling to get the specified number of bins
            pooled = F.adaptive_max_pool2d(x, (level, level))
            # Flatten the output for this level and add to our features
            features.append(pooled.view(batch_size, -1))
        
        # Concatenate all features from different levels
        return torch.cat(features, dim=1)


"""
CNN model with Spatial Pyramid Pooling to handle variable sized inputs while 
producing fixed-size feature vectors for character recognition
"""
class CNN:
    def __init__(self, cnn_params: CNNParams):
        self.cnn_params = cnn_params
        # Default SPP levels if not provided
        if 'spp_levels' not in cnn_params:
            self.cnn_params['spp_levels'] = [1, 2, 4]  # Default pyramid levels
        self.model = self.CNN2D(cnn_params)
        self.epoch_losses = []
        self.training_time = None
    
    class CNN2D(nn.Module):
        def __init__(self, cnn_params: CNNParams):
            super().__init__()
            # Define hyperparams
            conv_kernel_size = (3, 3)
            lrelu_neg_slope = 0.1

            # CNN layers - removed maxpooling from the original design to preserve more spatial information for SPP
            conv1_in, conv1_out = 1, 32
            conv2_in, conv2_out = conv1_out, 64
            conv3_in, conv3_out = conv2_out, 128
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv1_out),
                nn.LeakyReLU(lrelu_neg_slope),
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv2_out),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.MaxPool2d(kernel_size=(2, 2)),  # Reduced pooling
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=conv3_in, out_channels=conv3_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv3_out),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.MaxPool2d(kernel_size=(2, 2)),  # Reduced pooling
            )
            
            # Spatial Pyramid Pooling layer
            self.spp = SpatialPyramidPooling(cnn_params['spp_levels'])
            
            # Calculate the output dimension from SPP
            # For each level, we'll have (level * level * num_channels) features
            spp_output_dim = sum([level * level * conv3_out for level in cnn_params['spp_levels']])
            
            # Classification layers
            l1_in, l1_out = spp_output_dim, 512
            l2_in, l2_out = l1_out, 128
            l3_in, l3_out = l2_out, cnn_params['num_classes']
            
            self.fc = nn.Sequential(
                nn.Linear(l1_in, l1_out),
                nn.Dropout(0.3),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.Linear(l2_in, l2_out),
                nn.Linear(l3_in, l3_out)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x.float())
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.spp(x)  # Apply SPP to get fixed-size output regardless of input dimensions
            x = self.fc(x)
            return x
            
        # Visualization helper functions for explaining why the model works
        def get_feature_maps(self, x: torch.Tensor):
            """Extract feature maps from each layer to visualize what the network learns"""
            feature_maps = {}
            
            # Get feature maps from each convolutional block
            conv1_out = self.conv1(x.float())
            feature_maps['conv1'] = conv1_out
            
            conv2_out = self.conv2(conv1_out)
            feature_maps['conv2'] = conv2_out
            
            conv3_out = self.conv3(conv2_out)
            feature_maps['conv3'] = conv3_out
            
            # Also capture the SPP output
            spp_out = self.spp(conv3_out)
            feature_maps['spp'] = spp_out
            
            return feature_maps
    
    def train_model(self, X_train, y_train):
        print("Training model...")
        start_time = time.time()
        
        # Custom collate function to handle variable-sized inputs
        def custom_collate_fn(batch):
            # Process a batch of data with potentially different sizes
            images = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            return images, labels
        
        # Create a custom dataset that handles variable-sized inputs
        class VariableSizeDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]
        
        # Convert labels to integers if they're tensors
        if isinstance(y_train, list) and all(isinstance(y, torch.Tensor) for y in y_train):
            y_train = [y.item() for y in y_train]
        elif isinstance(y_train, torch.Tensor) and y_train.dim() == 1:
            y_train = y_train.tolist()

        # Split data for validation
        val_size = int(0.1 * len(y_train))
        indices = torch.randperm(len(y_train)).tolist()
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        if isinstance(X_train, list):
            X_train_split = [X_train[i] for i in train_indices]
            y_train_split = [y_train[i] for i in train_indices]
            X_val = [X_train[i] for i in val_indices]
            y_val = [y_train[i] for i in val_indices]
        else:
            X_train_split = X_train[train_indices]
            y_train_split = [y_train[i] for i in train_indices]
            X_val = X_train[val_indices]
            y_val = [y_train[i] for i in val_indices]
        
        # Create datasets
        train_dataset = VariableSizeDataset(X_train_split, y_train_split)
        val_dataset = VariableSizeDataset(X_val, y_val)
        
        # Create dataloaders with custom collate function
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=self.cnn_params["batch_size"],
            collate_fn=custom_collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.cnn_params["batch_size"],
            collate_fn=custom_collate_fn
        )

        # Define optimizer, scheduler, and loss function
        optimiser = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.cnn_params["learning_rate"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=3, verbose=True
        )
        loss_fn = nn.CrossEntropyLoss()
        
        # Training metrics to analyse
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Track best model
        best_val_accuracy = 0
        best_model_state = None
        
        for i in range(self.cnn_params["num_epochs"]):
            self.model.train()
            epoch_train_loss = 0
            train_predictions = []
            train_targets = []
            
            for batch_data in train_loader:
                x_batch, y_batch = batch_data
                optimiser.zero_grad()
                
                # Process each image in the batch individually since they may have different sizes
                batch_outputs = []
                for x in x_batch:
                    # Add batch dimension if not already present
                    if len(x.shape) == 3:
                        x = x.unsqueeze(0)
                    # Forward pass
                    output = self.model(x)
                    batch_outputs.append(output)
                
                # Stack the outputs from individual images
                y_pred = torch.cat(batch_outputs, dim=0)
                
                # Compute loss and backpropagate
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimiser.step()
                
                epoch_train_loss += loss.item()
                train_predictions.extend(torch.argmax(y_pred, dim=1).detach().cpu().numpy())
                train_targets.extend(y_batch.cpu().numpy())

            epoch_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_targets, train_predictions)
            train_losses.append(epoch_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    batch_outputs = []
                    for x in x_batch:
                        if len(x.shape) == 3:
                            x = x.unsqueeze(0)
                        output = self.model(x)
                        batch_outputs.append(output)
                    
                    val_output = torch.cat(batch_outputs, dim=0)
                    val_loss = loss_fn(val_output, y_batch)
                    epoch_val_loss += val_loss.item()
                    val_predictions.extend(torch.argmax(val_output, dim=1).cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())
            
            epoch_val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Calculate per-class metrics to understand model behavior
            if i % 5 == 0 or i == self.cnn_params["num_epochs"] - 1:
                class_precision = precision_score(val_targets, val_predictions, average=None, zero_division=0)
                class_recall = recall_score(val_targets, val_predictions, average=None, zero_division=0)
                top5_worst_classes = np.argsort(class_precision)[:5]
                print(f"Five worst-performing classes: {[CHARACTERS[idx] for idx in top5_worst_classes]}")
                print(f"Their precision: {class_precision[top5_worst_classes]}")
                print(f"Their recall: {class_recall[top5_worst_classes]}")
            
            print(f"Epoch {i+1}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            self.epoch_losses.append(epoch_train_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(epoch_val_loss)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                print(f"New best model with validation accuracy: {val_accuracy:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored final best model with validation accuracy: {best_val_accuracy:.4f}")
        
        self.epoch_losses = train_losses
        self.training_time = time.time() - start_time

    def save_trained_model(self, model_path: str):
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save((self.model.state_dict(), self.epoch_losses, self.training_time), model_path)
        print("Saved trained model to cache.")
    
    def load_trained_model(self, model_path: str):
        try:
            model_state_dict, epoch_losses, training_time = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)
            self.epoch_losses = epoch_losses
            self.training_time = training_time
            print(f"Trained model (took {int(self.training_time // 60)}m {int(self.training_time % 60)}s) has the saved epoch losses: ")
            for i, epoch_loss in enumerate(self.epoch_losses):
                print(f"Epoch {i+1}, Loss: {epoch_loss}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Will train a new model.")
            raise

    def predict(self, X):
        """Predict class labels for inputs X"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, list):
                preds = []
                for x in X:
                    if len(x.shape) == 3:  # Add batch dimension if missing
                        x = x.unsqueeze(0)
                    output = self.model(x)
                    preds.append(torch.argmax(output, dim=1).item())
                return preds
            else:
                if len(X.shape) == 3:  # Add batch dimension if missing
                    X = X.unsqueeze(0)
                output = self.model(X)
                return torch.argmax(output, dim=1)
                
    def predict_captcha(self, X_test_captcha):
        """
        Predict an entire CAPTCHA by processing each character image
        Args:
            X_test_captcha: List of lists, where each inner list contains character images for a CAPTCHA
        Returns:
            List of predicted CAPTCHA strings
        """
        self.model.eval()
        captcha_predictions = []
        
        with torch.no_grad():
            for captcha_images in X_test_captcha:
                char_predictions = []
                for char_img in captcha_images:
                    if len(char_img.shape) == 3:
                        char_img = char_img.unsqueeze(0)
                    output = self.model(char_img)
                    pred_idx = torch.argmax(output, dim=1).item()
                    char_predictions.append(CHARACTERS[pred_idx])
                
                captcha_predictions.append(''.join(char_predictions))
                
        return captcha_predictions