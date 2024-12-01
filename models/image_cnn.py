"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # Image processing

# ------------------------------------------------------------------------------------- #
# Classes
# ------------------------------------------------------------------------------------- #

class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from directories.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing subdirectories for each class.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load all image paths and labels
        for idx, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for image_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, image_file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(image_path) as img:
            img = img.convert("L")  # Convert to grayscale
            if self.transform:
                img = self.transform(img)
        return img, label

class LeNet5(nn.Module):
    """
    LeNet-5 architecture with dynamic computation of the fully connected input size.
    """
    def __init__(self, num_classes=10, input_size=(32, 32)):
        super(LeNet5, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # Conv1
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Pool1
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # Conv2
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # Pool2
            nn.Conv2d(16, 120, kernel_size=5, stride=1),  # Conv3
            nn.Tanh(),
        )

        # Dynamically compute the input size for the first fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)  # Simulate input with given size
            self.fc_input_size = self.feature_extractor(dummy_input).view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 84),  # FC1
            nn.Tanh(),
            nn.Linear(84, num_classes),  # FC2
        )

    def forward(self, x):
        # Pass through feature extractor
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten for the classifier
        # Pass through classifier
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Train Epoch {epoch}: Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * correct / len(train_loader.dataset):.2f}%")

def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Test: Loss: {total_loss / len(test_loader):.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%")

# ------------------------------------------------------------------------------------- #
# Main Execution
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to datasets
    train_path = "data/combined_by_type1/train"
    val_path = "data/combined_by_type1/val"
    test_path = "data/combined_by_type1/test"

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    num_classes = len(os.listdir(train_path))  # Number of classes based on directories

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match LeNet-5 input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Datasets and loaders
    train_dataset = CustomImageDataset(root_dir=train_path, transform=transform)
    val_dataset = CustomImageDataset(root_dir=val_path, transform=transform)
    test_dataset = CustomImageDataset(root_dir=test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, optimizer
    model = LeNet5(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        print("Validation Performance:")
        test(model, device, val_loader, criterion)

    # Final test evaluation
    print("Final Test Performance:")
    test(model, device, test_loader, criterion)

    # Save the model
    torch.save(model.state_dict(), "lenet5_custom.pth")
    print("Model saved to lenet5_custom.pth")