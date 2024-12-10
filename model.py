import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Define the ImprovedBankMarketingModel class which inherits from nn.Module
class ImprovedBankMarketingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the model.
        Parameters:
        - input_dim: Number of input features (dimensionality of the input).
        - hidden_dim: Number of neurons in the hidden layers.
        - output_dim: Number of output classes (2 for binary classification).
        """
        super(ImprovedBankMarketingModel, self).__init__()
        # Fully connected layers (Linear layers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer to prevent overfitting
 
    def forward(self, x):
        """
        Define the forward pass through the network.
        Parameters:
        - x: Input tensor (batch of data).
        """
        x = self.relu(self.fc1(x))  # Pass through first layer and apply ReLU activation
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.relu(self.fc2(x))  # Pass through second layer and apply ReLU activation
        x = self.fc3(x)  # Output layer (no activation function, logits for classification)
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Train the model using the provided data loader, loss function, and optimizer.
    Parameters:
    - model: The neural network model.
    - train_loader: DataLoader providing the training data in batches.
    - criterion: The loss function.
    - optimizer: Optimizer used to update model weights.
    - epochs: Number of training epochs.
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):  # Loop over each epoch
        running_loss = 0.0  # Initialize loss accumulator
        for inputs, labels in train_loader:  # Loop over each batch
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model's weights
            running_loss += loss.item()  # Accumulate loss for logging
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}")  # Print loss after each epoch

# Function to perform weighted soft voting for ensemble models
def weighted_soft_voting(models, weights, test_x):
    """
    Perform weighted soft voting for an ensemble of models.
    Parameters:
    - models: List of trained models.
    - weights: List of weights for each model's contribution to the final prediction.
    - test_x: Test data to evaluate the models on.
    """
    model_probs = []  # List to store probabilities from each model
    with torch.no_grad():  # No gradient computation for evaluation
        for model in models:
            model.eval()  # Set the model to evaluation mode
            outputs = model(test_x)  # Get the model outputs
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]  # Convert logits to probabilities for class 1
            model_probs.append(probs.numpy())  # Store the probabilities as numpy arrays

    # Weighted average of probabilities from all models in the ensemble
    ensemble_probs = np.average(np.column_stack(model_probs), axis=1, weights=weights)
    return ensemble_probs  # Return the final ensemble probabilities

# Define the FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        """
        Focal Loss to address class imbalance by focusing on hard-to-classify examples.
        Parameters:
        - alpha: Balancing factor for the loss. Can be used to adjust the importance of different classes.
        - gamma: Focusing parameter to reduce easy-to-classify examples' loss.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Set the balancing factor
        self.gamma = gamma  # Set the focusing parameter

    def forward(self, inputs, targets):
        """
        Compute the focal loss.
        Parameters:
        - inputs: Model predictions (logits).
        - targets: Ground truth labels (indices of correct class).
        """
        # Compute the cross-entropy loss for each sample (without reduction to obtain per-sample losses)
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        
        # Compute the probability of the true class for each sample
        pt = torch.exp(-ce_loss)  # pt is the probability for the correct class
        
        # Compute the focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Return the mean of the focal loss across all samples
        return focal_loss.mean()
