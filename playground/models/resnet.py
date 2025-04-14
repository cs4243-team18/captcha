import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import TypedDict, Tuple
from helper_functions.data_transformation import CHARACTERS, IMG_HEIGHT, IMG_WIDTH
from helper_functions.extract_features import NUM_ENGINEERED_FEATURES

class CNNParams(TypedDict):
    num_classes: int
    input_shape: Tuple[int, int, int] # (channels, height, width)
    learning_rate: float
    num_epochs: int
    batch_size: int

class HybridCNNParams(TypedDict):
    num_classes: int
    num_engineered_features: int
    input_shape: Tuple[int, int, int] # (channels, height, width)
    learning_rate: float
    num_epochs: int
    batch_size: int

class HybridDataset(Dataset):
    def __init__(self, X_img, X_features, y):
        self.X_img = torch.tensor(X_img, dtype=torch.float32)
        self.X_features = torch.tensor(X_features, dtype=torch.float32)
        self.y = y

    def __len__(self):
        return len(self.X_img)

    def __getitem__(self, idx):
        return (self.X_img[idx], self.X_features[idx]), self.y[idx]
    
class ResNet50(nn.Module):
    def __init__(self, cnn_params: CNNParams, pretrained_weights='IMAGENET1K_V1'):
        super(ResNet50, self).__init__()
        self.cnn_params = cnn_params
        
        # Initialize ResNet50 with pretrained weights (optional)
        self.resnet = models.resnet50(weights=pretrained_weights)

        # Modify output layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of input features to the FC layer
        self.resnet.fc = nn.Linear(num_ftrs, cnn_params['num_classes'])  # Replace FC layer with 128 output units

        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
 
    def forward(self, x):
        return self.resnet(x)

    def train_model(self, X_train, y_train):
        self.train()  # Set to training mode

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(self.parameters(), lr=self.cnn_params['learning_rate'])  # Adam optimizer
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.cnn_params["batch_size"])
        for epoch in tqdm(range(self.cnn_params['num_epochs'])):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients for the optimizer
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                running_loss += loss.item()  # Accumulate the loss

            print(f"Epoch [{epoch+1}/{self.cnn_params['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}")

        print("Training complete!")


class HybridResNet50(nn.Module):
    def __init__(self, hybrid_cnn_params: HybridCNNParams, pretrained_weights='IMAGENET1K_V1'):
        super(HybridResNet50, self).__init__()
        self.hybrid_cnn_params = hybrid_cnn_params
        # Initialize ResNet50 with pretrained weights (optional)
        self.resnet = models.resnet50(weights=pretrained_weights)

        # Modify output layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of input features to the FC layer
        self.resnet.fc = nn.Identity()  # Replace FC layer with 128 output units

        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        resnet_output_dim = 2048

        self.fc_engineered = nn.Sequential(
            nn.Linear(hybrid_cnn_params['num_engineered_features'], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(resnet_output_dim + 64, 256),  # 512 (ResNet) + 64 (engineered)
            nn.ReLU(),
            nn.Linear(256, hybrid_cnn_params['num_classes'])
        )
 
    def forward(self, x, x_features):
        x = self.resnet(x)
        x_features = self.fc_engineered(x_features)
        combined_features = torch.cat((x, x_features), dim=1)
        print(combined_features.shape)
        return self.final_fc(combined_features)

    def train_model(self, X_train,  X_features, y_train):
        self.train()  # Set to training mode

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer
        dataset = HybridDataset(X_train, X_features, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.cnn_params["batch_size"])
        for epoch in range(self.hybrid_cnn_params['learning_rate']):
            running_loss = 0.0
            for (images, features), labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients for the optimizer
                outputs = self(images, features)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                running_loss += loss.item()  # Accumulate the loss

            print(f"Epoch [{epoch+1}/{self.hybrid_cnn_params['num_epochs']}], Loss: {running_loss/len(train_loader):.4f}")

        print("Training complete!")
