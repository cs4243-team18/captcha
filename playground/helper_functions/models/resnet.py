import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from ..data_transformation import CHARACTERS, IMG_HEIGHT, IMG_WIDTH
from ..extract_features import NUM_ENGINEERED_FEATURES

class ResNet50(nn.Module):
    def __init__(self, input_size = (IMG_HEIGHT, IMG_WIDTH), output_size = len(CHARACTERS), pretrained_weights='IMAGENET1K_V1'):
        super(ResNet50, self).__init__()

        # Initialize ResNet50 with pretrained weights (optional)
        self.resnet = models.resnet50(weights=pretrained_weights)

        # Modify output layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of input features to the FC layer
        self.resnet.fc = nn.Linear(num_ftrs, output_size)  # Replace FC layer with 128 output units

        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
 
    def forward(self, x):
        return self.resnet(x)

    def train_model(self, train_loader, num_epochs=100):
        self.train()  # Set to training mode

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients for the optimizer
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                running_loss += loss.item()  # Accumulate the loss

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        print("Training complete!")

    def evaluate(self, test_loader):
        self.eval()  # Set to evaluation mode
        
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():  # No need to track gradients during evaluation
            for images, labels in test_loader:
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                running_loss += loss.item()  # Accumulate the loss

                # Get the predicted class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {running_loss/len(test_loader):.4f}")
        print(f"Accuracy: {100 * correct / total:.2f}%")

class HybridResNet50(nn.Module):
    def __init__(self, input_size = (IMG_HEIGHT, IMG_WIDTH), output_size = len(CHARACTERS), num_engineered_features = NUM_ENGINEERED_FEATURES, pretrained_weights='IMAGENET1K_V1'):
        super(HybridResNet50, self).__init__()

        # Initialize ResNet50 with pretrained weights (optional)
        self.resnet = models.resnet50(weights=pretrained_weights)

        # Modify output layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of input features to the FC layer
        self.resnet.fc = nn.Identity()  # Replace FC layer with 128 output units

        # Modify first conv layer
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        resnet_output_dim = 2048

        self.fc_engineered = nn.Sequential(
            nn.Linear(num_engineered_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(resnet_output_dim + 64, 256),  # 512 (ResNet) + 64 (engineered)
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
 
    def forward(self, x, x_features):
        x = self.resnet(x)
        x_features = self.fc_engineered(x_features)
        combined_features = torch.cat((x, x_features), dim=1)
        print(combined_features.shape)
        return self.final_fc(combined_features)

    def train_model(self, train_loader, num_epochs=100):
        self.train()  # Set to training mode

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adam optimizer

        for epoch in range(num_epochs):
            running_loss = 0.0
            for (images, features), labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients for the optimizer
                outputs = self(images, features)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model parameters

                running_loss += loss.item()  # Accumulate the loss

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        print("Training complete!")

    def evaluate(self, test_loader):
        self.eval()  # Set to evaluation mode
        
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():  # No need to track gradients during evaluation
            for (images, features), labels in test_loader:
                outputs = self(images, features)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                running_loss += loss.item()  # Accumulate the loss

                # Get the predicted class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Loss: {running_loss/len(test_loader):.4f}")
        print(f"Accuracy: {100 * correct / total:.2f}%")
