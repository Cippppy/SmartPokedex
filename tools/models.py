"""
## Summary
This script defines a Convolutional Neural Network (CNN) model for image classification tasks.
The CNNClassifier class can be used for classifying images into specified classes based on
convolutional and fully connected layers, with dropout for regularization.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch  # PyTorch library for tensor operations and deep learning
import torch.nn as nn  # Neural network modules in PyTorch

# ------------------------------------------------------------------------------------- #
# Classes
# ------------------------------------------------------------------------------------- #

class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network (CNN) model for image classification.
    
    This model includes three convolutional layers followed by max pooling, and two fully connected layers.
    A dropout layer is included after the first fully connected layer for regularization.
    
    Args:
        num_classes (int): The number of output classes for the classifier.
        
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 3 input channels (RGB) and 32 output channels.
        conv2 (nn.Conv2d): Second convolutional layer with 32 input channels and 64 output channels.
        conv3 (nn.Conv2d): Third convolutional layer with 64 input channels and 128 output channels.
        pool (nn.MaxPool2d): Max pooling layer with a 2x2 window.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer (final layer with num_classes outputs).
        dropout (nn.Dropout): Dropout layer with 50% dropout rate.
    """
    
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        
        # Define convolutional layers with ReLU activations and padding to retain spatial dimensions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # For RGB images, 3 input channels
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer to reduce spatial size by half
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Adjusted for image size (assuming 64x64 input size)
        self.fc2 = nn.Linear(256, num_classes)  # Output layer with num_classes nodes for classification
        
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).
            
        Returns:
            torch.Tensor: Output logits for each class, with shape (batch_size, num_classes).
        """
        # Apply first convolution, ReLU activation, and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        
        # Apply second convolution, ReLU activation, and pooling
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Apply third convolution, ReLU activation, and pooling
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 8 * 8)  # Adjust if input size is different
        
        # Apply first fully connected layer with ReLU and dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply final fully connected layer to get logits for each class
        x = self.fc2(x)
        
        return x
