# Import packages
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn 
import time


class CNN:     
    def __init__(self):
        self.model = self.CNN2D(6)
        self.epoch_losses = []
    
    def fit(self, X, y):
        """
        Train the model using the input data.
        
        Parameters
        ----------
        X : list of n CAPTCHA images
        y : list of n CAPTCHA labels
            
        Returns
        -------
        self : object
            Returns an instance of the trained model.
        """
        # Preprocess and segment the input CAPTCHA images to obtain (new_X, new_y)
        # of input character images and their corresponding character labels, 
        # making sure to skip incorrectly segmented images.
        new_X, new_y = [], []



        

        # Now, t]
        X = self.transform_data(X)
        X, y = torch.tensor(X), torch.tensor(y)
        self.train_model(X, y)
        return self.model
    
    def predict(self, X):
        """
        Use the trained model to make predictions.
        
        Parameters
        ----------
        X : list of size (n_samples)
            Each item in the list is a grayscale video of shape (L, H, W).
            L represents the length of the video, which may vary between videos. 
            H and W represent the height and width, which are consistent across all videos. 
            
        Returns
        -------
        ndarray of shape (n_samples,)
        Predicted target values per element in X.
           
        """
        X = self.transform_data(X)
        X = torch.tensor(X)
        predictions = torch.argmax(self.model(X), axis=1)
        # print("WE MADE IT")
        return predictions
    
    def transform_data(self, X):
        transformed_X = self.replace_missing_vals_and_outliers(X)
        # Change all (x,16,16) tensors to (10,16,16) tensors
        transformed_X = self.pad_frames(transformed_X)
        return transformed_X
    
    def replace_missing_vals_and_outliers(self, X):
        n = len(X)
        cleaned_X = [None] * n
        for i in range(len(X)):
            video = X[i]
            new_video = []
            for frame in video:
                frame_flat = frame.flatten()
                Q2 = np.nanpercentile(frame_flat, 50)
                frame_flat = np.nan_to_num(frame_flat, Q2)
                frame_flat = np.where((frame_flat < 0) | (frame_flat > 255), Q2, frame_flat)
                new_video.append(frame_flat.reshape(16, 16))
            cleaned_X[i] = np.array(new_video) / 255
        return cleaned_X

    def pad_frames(self, X_train):
        n = len(X_train)
        processed_X = np.zeros((n, 10, 16, 16))
        for i in range(n):
            video = X_train[i]
            num_frames = video.shape[0]
            # If alr 10 frames continue, else pad with the last frame
            if num_frames == 10:
                processed_X[i] = video
            else:
                # Basically duplicate the last few frames in order to match 10 frames
                num_to_add = 10 - num_frames
                latest_frames = video[-num_to_add:]
                pad = [frame for frame in latest_frames for _ in range(2)]
                processed_X[i] = np.concatenate((video[:10-num_to_add*2], pad), axis=0)
        return processed_X
    
    def oversample(self, X_train, y_train):
        classes = np.unique(y_train)
        class_count = {}
        for c in y_train:
            if c not in class_count: 
                class_count[c] = 1
            else: 
                class_count[c] += 1

        oversample_size = int(max(max(class_count.values()) / 3, min(class_count.values()) * 2))
        oversampled_X = X_train
        oversampled_y = y_train

        for c in classes:
            diff = oversample_size - class_count[c] 
            if diff <= 0: continue
            class_indices = np.where(y_train == c)[0]
            oversample_indices = np.random.choice(class_indices, size=diff, replace=True)
            oversampled_X = np.concatenate((oversampled_X, X_train[oversample_indices]), axis=0)
            oversampled_y = np.concatenate((oversampled_y, np.full(diff, c, dtype=y_train.dtype)), axis=0)

        return oversampled_X, oversampled_y


    class CNN2D(nn.Module):
        def __init__(self, classes):
            super().__init__()
            # First define all layer and activation function sizes and params
            conv1_in, conv1_out = 10, 32
            conv2_in, conv2_out = conv1_out, 64
            l1_in, l1_out = 256, 128
            l2_in, l2_out = l1_out, classes
            # l2_in, l2_out = l1_out, 64
            l3_in, l3_out = l2_out, classes

            conv_kernel_size = (3,3)
            maxpool_kernel_size = (2,2)
            lrelu_neg_slope = 0.1

            # CNN layer
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=conv_kernel_size),
                nn.BatchNorm2d(conv1_out),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=conv_kernel_size),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(kernel_size=maxpool_kernel_size),
                nn.LeakyReLU(lrelu_neg_slope)
            )

            # Classification time
            self.fc = nn.Sequential(
                nn.Linear(l1_in, l1_out),
                nn.Dropout(0.3),
                nn.LeakyReLU(lrelu_neg_slope),
                nn.Linear(l2_in, l2_out),
                # nn.LeakyReLU(lrelu_neg_slope),
                # nn.Linear(l3_in, l3_out)
            )

        def forward(self, x):
            # Once again, follow the given architecture as mentioned above.
            x = x.float()
            # CNN
            x = self.conv(x)
            # print(f"OK TIME TO FLATTEN: {x.shape}")
            a,b,c,d = x.shape
            x = x.view(-1, b*c*d) 
            # print(f"AFTER FLATTENING: {x.shape}")
            # Classification time
            x = self.fc(x)
            # Output should be 2D (batch, 6)
            return x
        
    def train_model(self, X_train, y_train):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=1.1*1e-3)
        loss_fn = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=250)
        for i in range(25):
            epoch_loss = 0
            for idx, data in enumerate(train_loader): # Each batch depends on loader size
                x, y = data
                # print(f"OK COME. BATCH GOING OUT NOW: {x.shape}, {y.shape}")
                # Reset optimiser's gradient
                optimiser.zero_grad()
                # Forward pass to get y_pred
                y_pred = self.model(x)
                # print(f"OK LETS SEE! y_pred has a shape of {y_pred.shape}")
                y = y.long()
                # Calculate loss
                loss = loss_fn(y_pred, y)
                # Backpropagate
                loss.backward()
                # Update model weights
                optimiser.step()

                epoch_loss += loss

            epoch_loss = epoch_loss / len(train_loader) # Get avg loss in cur batch
            self.epoch_losses.append(epoch_loss.item())


# Load data
with open('data.npy', 'rb') as f:
    data = np.load(f, allow_pickle=True).item()
    X = data['data']
    y = data['label']

