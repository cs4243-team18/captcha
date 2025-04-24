# Import packages
import time
import numpy as np
import torch
import torch.nn as nn 
from typing import TypedDict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np

from helper_functions.data_transformation import CHARACTERS

class CNNParams(TypedDict):
    num_classes: int
    input_shape: Tuple[int, int, int] # (channels, height, width)
    learning_rate: float
    num_epochs: int
    batch_size: int


"""
CNN model with 5 convolutional layers and 3 fc layers to recognise characters 
"""
class CNN:
    def __init__(self, cnn_params: CNNParams):
        self.cnn_params = cnn_params
        self.model = self.CNN2D(cnn_params)
        self.epoch_losses = []
        self.training_time = None
    
    class CNN2D(nn.Module):
        def __init__(self, cnn_params: CNNParams):
            super().__init__()
            # Define hyperparams
            conv_kernel_size = (3,3)
            maxpool_kernel_size = (2,2)
            lrelu_neg_slope = 0.1

            # CNN layers
            conv1_in, conv1_out = 1, 32
            conv2_in, conv2_out = conv1_out, 64
            conv3_in, conv3_out = conv2_out, 96
            conv4_in, conv4_out = conv3_out, 128
            conv5_in, conv5_out = conv4_out, 160
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv1_out),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv2_out),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),
            )
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=conv3_in, out_channels=conv3_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv3_out),
                nn.LeakyReLU(lrelu_neg_slope),
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=conv4_in, out_channels=conv4_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv4_out),
                nn.LeakyReLU(lrelu_neg_slope),
            )

            self.conv5 = nn.Sequential(
                nn.Conv2d(in_channels=conv5_in, out_channels=conv5_out, kernel_size=conv_kernel_size, padding=1),
                nn.BatchNorm2d(conv5_out),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),
            )

            # Classification layers
            dummy_conv_input = torch.randn((1,) + cnn_params['input_shape'])
            dummy_conv1_output = self.conv1(dummy_conv_input)
            dummy_conv2_output = self.conv2(dummy_conv1_output)
            dummy_conv3_output = self.conv3(dummy_conv2_output)
            dummy_conv4_output = self.conv4(dummy_conv3_output)
            dummy_conv5_output = self.conv5(dummy_conv4_output)
            flattened_shape = dummy_conv5_output.numel() // dummy_conv5_output.size(0)
            l1_in, l1_out = flattened_shape, 512
            l2_in, l2_out = l1_out, 128
            l3_in, l3_out = l2_out, cnn_params['num_classes']
            
            self.fc = nn.Sequential(
                nn.Linear(l1_in, l1_out),
                nn.Dropout(0.3),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.Linear(l2_in, l2_out),
                nn.Dropout(0.3),
                nn.Linear(l3_in, l3_out)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x.float())
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = torch.flatten(x, start_dim=1)
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
            
            return feature_maps
    
    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor):
        print("Training model...")
        start_time = time.time()
        
        # Split data for validation - helps explain generalization
        val_size = int(0.1 * len(y_train))
        indices = torch.randperm(len(y_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # Create datasets and dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=self.cnn_params["batch_size"]
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.cnn_params["batch_size"]
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
                x, y = batch_data
                optimiser.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimiser.step()
                
                epoch_train_loss += loss.item()
                train_predictions.extend(torch.argmax(y_pred, dim=1).detach().cpu().numpy())
                train_targets.extend(y.cpu().numpy())

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
                for x_val, y_val in val_loader:
                    val_output = self.model(x_val)
                    val_loss = loss_fn(val_output, y_val)
                    epoch_val_loss += val_loss.item()
                    val_predictions.extend(torch.argmax(val_output, dim=1).cpu().numpy())
                    val_targets.extend(y_val.cpu().numpy())
            
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
        torch.save((self.model.state_dict(), self.epoch_losses, self.training_time), model_path)
        print("Saved trained model to cache.")
    
    def load_trained_model(self, model_path: str):
        model_state_dict, epoch_losses, training_time = torch.load(model_path)
        self.model.load_state_dict(model_state_dict)
        self.epoch_losses = epoch_losses
        self.training_time = training_time
        print(f"Trained model (took {int(self.training_time // 60)}m {int(self.training_time % 60)}s) has the saved epoch losses: ")
        for i, epoch_loss in enumerate(self.epoch_losses):
            print(f"Epoch {i+1}, Loss: {epoch_loss}")