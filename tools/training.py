"""
## Summary
This script contains functions for training various machine learning models, including linear regression,
Ridge/Lasso regression with hyperparameter optimization, YOLO for object detection, a CNN for image classification,
and KNN for image-based classification.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch  # PyTorch for deep learning models
import pandas as pd  # Data manipulation and analysis library
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Linear regression models
from sklearn.model_selection import GridSearchCV  # Hyperparameter tuning
from sklearn.metrics import mean_squared_error  # Evaluation metric
from ultralytics import YOLO  # YOLO model for object detection
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def train_basic_regression(train_df: pd.DataFrame):
    """
    Train a basic linear regression model using training data.
    
    Args:
        train_df (pd.DataFrame): DataFrame containing training data with features and target.
        
    Returns:
        model: Trained linear regression model.
    """
    X_train = train_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_train = train_df['Total']

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def train_optimized_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, model_type="ridge"):
    """
    Train a Ridge or Lasso regression model with hyperparameter tuning on validation data.
    
    Args:
        train_df (pd.DataFrame): DataFrame for training data.
        val_df (pd.DataFrame): DataFrame for validation data.
        model_type (str): Type of model to train ('ridge' or 'lasso').
        
    Returns:
        best_model: The best trained model based on grid search.
    """
    # Extract features and target from the training dataset
    X_train = train_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_train = train_df['Total']
    
    # Extract features and target from the validation dataset
    X_val = val_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_val = val_df['Total']

    # Choose model and set up hyperparameter grid for tuning
    if model_type == "ridge":
        model = Ridge()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == "lasso":
        model = Lasso()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    else:
        raise ValueError("model_type must be either 'ridge' or 'lasso'")

    # Grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best model and evaluate on the validation set
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Validation MSE: {val_mse}")

    return best_model

def train_yolo(model_path: str):
    """
    Train a YOLO model on Pokémon image data for classification.
    
    Args:
        model_path (str): Path to the pretrained YOLO model or model configuration file.
        
    Returns:
        results: Training results from the YOLO model.
    """
    # Load the YOLO model
    model = YOLO(model_path)  # Load a pretrained YOLO model or model yaml file

    # Train the model on the Pokémon image dataset with specified settings
    results = model.train(data="kaggle/pokemon_by_type1/", epochs=20, imgsz=128, device='cuda')

    return results

def train_image_cnn(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cuda'):
    """
    Train a Convolutional Neural Network (CNN) for image classification on Pokémon image data.
    
    Args:
        model: The CNN model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        optimizer: Optimization algorithm.
        criterion: Loss function.
        epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
    """
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient calculations for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%")

def train_image_knn(X_train, y_train, k):
    """
    Train a K-Nearest Neighbors (KNN) classifier for image classification.
    
    Args:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        k (int): Number of neighbors to use in the KNN model.
        
    Returns:
        knn: Trained KNN model.
    """
    # Initialize the KNN classifier with specified number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN classifier
    knn.fit(X_train, y_train)

    return knn

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Run YOLO training as a standalone example
    train_yolo("yolo11n-cls.pt")
