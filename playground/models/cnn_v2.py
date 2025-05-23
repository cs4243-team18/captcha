# Import packages
import numpy as np
import torch
import torch.nn as nn 
from typing import TypedDict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np

class CNNParams(TypedDict):
    num_classes: int
    input_shape: Tuple[int, int, int] # (channels, height, width)
    learning_rate: float
    num_epochs: int
    batch_size: int


"""
Basic CNN model to recognise characters.
Note: This model does not handle segmentation of CAPTCHA images into individual characters; it only trains on and 
recognises characters after segmentation.
"""
class CNN:     
    def __init__(self, cnn_params: CNNParams):
        self.cnn_params = cnn_params
        self.model = self.CNN2D(cnn_params)
        self.epoch_losses = []
    
    # Underlying 2D CNN architecture
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
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=conv_kernel_size),
                nn.BatchNorm2d(conv1_out),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),

                nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=conv_kernel_size),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),
            )

            # Classification layers
            dummy_conv_input = torch.randn((1,) + cnn_params['input_shape']) # Input shape of (N,C,H,W)
            dummy_conv_output = self.conv(dummy_conv_input) # Output shape of (N,C2,H2,W2)
            flattened_shape = dummy_conv_output.numel() // dummy_conv_output.size(0) # Get C2*H2*W2
            l1_in, l1_out = flattened_shape, 128
            l2_in, l2_out = l1_out, cnn_params['num_classes']
            self.fc = nn.Sequential(
                nn.Linear(l1_in, l1_out),
                nn.Dropout(0.3),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.Linear(l2_in, l2_out),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x.float()) # Ensure float for PyTorch
            # Flatten output of the conv layer for fc layer, i.e. convert (N,C,H,W) to (N,C*H*W)
            x = torch.flatten(x, start_dim=1) 
            x = self.fc(x)
            return x
    
    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Trains the model using the input data.
        
        Parameters
        X_train: tensor of n training character images, where each grayscale character image is a normalised matrix 
            of shape (IMG_HEIGHT, IMG_WIDTH)
        y_train: tensor of n character labels (dneoted by integers)
        """
        print("Training model...")
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.cnn_params["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.cnn_params["batch_size"])
        for i in range(self.cnn_params["num_epochs"]):
            epoch_loss = 0
            for batch_data in train_loader:
                x, y = batch_data
                optimiser.zero_grad() # Reset optimiser's gradient
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimiser.step() # Update model weights
                epoch_loss += loss

            epoch_loss = epoch_loss / len(train_loader) # Get avg loss so far
            print(f"Epoch {i+1}, Loss: {epoch_loss.item()}")
            self.epoch_losses.append(epoch_loss.item())

    def save_trained_model(self, model_path: str):
        torch.save((self.model.state_dict(), self.epoch_losses), model_path)
        print("Saved trained model to cache.")
    
    def load_trained_model(self, model_path: str):
        model_state_dict, epoch_losses = torch.load(model_path)
        self.model.load_state_dict(model_state_dict)
        self.epoch_losses = epoch_losses
        print("Loaded trained model from cache with the saved epoch losses:")
        for i, epoch_loss in enumerate(self.epoch_losses):
            print(f"Epoch {i+1}, Loss: {epoch_loss}")