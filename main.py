"""
Smart Pokedex

## Summary
Main script for training and evaluating various machine learning models (e.g., regression, KNN, CNN)
on Pokémon-related data, specifically using PyTorch for deep learning tasks.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch.nn as nn  # Neural network modules in PyTorch
import torch.optim as optim  # Optimization algorithms in PyTorch
import pandas as pd  # Data manipulation and analysis library
from ultralytics import YOLO

# Local imports - Custom functions and models for specific tasks
from tools.models import CNNClassifier  # CNN model class for image classification tasks
from tools.preprocess import split_csv_data, preprocess_image_knn, preprocess_image_cnn  # Preprocessing functions
from tools.training import train_basic_regression, train_yolo, train_optimized_regression, train_image_knn, train_image_cnn  # Training functions
from tools.predicting import eval_basic_regression, eval_image_knn, eval_image_cnn  # Evaluation functions

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def basic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train and evaluate a basic regression model on the training and test datasets.
    
    Args:
        train_df (pd.DataFrame): Training data as a DataFrame.
        test_df (pd.DataFrame): Test data as a DataFrame.
        
    Returns:
        tuple: Mean Squared Error (mse) and R^2 score (r2) of the model on the test dataset.
    """
    # Train the regression model
    model = train_basic_regression(train_df)
    # Evaluate the model and calculate metrics
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

def optimized_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train and evaluate an optimized regression model on the training, validation, and test datasets.
    
    Args:
        train_df (pd.DataFrame): Training data as a DataFrame.
        val_df (pd.DataFrame): Validation data for tuning as a DataFrame.
        test_df (pd.DataFrame): Test data for final evaluation as a DataFrame.
        
    Returns:
        tuple: Mean Squared Error (mse) and R^2 score (r2) of the model on the test dataset.
    """
    # Train the optimized regression model using the training and validation data
    model = train_optimized_regression(train_df, val_df)
    # Evaluate the model and calculate metrics
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

def image_knn(k, train_path, val_path, test_path):
    """
    Train and evaluate a K-Nearest Neighbors (KNN) classifier on image data.
    
    Args:
        k (int): Number of neighbors to consider in the KNN model.
        train_path (str): Directory path to training images.
        val_path (str): Directory path to validation images.
        test_path (str): Directory path to test images.
    """
    # Preprocess the images for KNN
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocess_image_knn(train_path, val_path, test_path)
    # Train the KNN model with the training data and specified k value
    model = train_image_knn(X_train, y_train, k)
    # Evaluate the model using validation and test data
    eval_image_knn(model, k, X_val, y_val, X_test, y_test, label_encoder)
    
def image_cnn(train_path, val_path, test_path, device='cuda'):
    """
    Train and evaluate a Convolutional Neural Network (CNN) on image data.
    
    Args:
        train_path (str): Directory path to training images.
        val_path (str): Directory path to validation images.
        test_path (str): Directory path to test images.
        device (str, optional): Device to run the model on ('cuda' for GPU, 'cpu' for CPU). Defaults to 'cuda'.
    """
    # Preprocess the images for CNN training
    train_loader, val_loader, test_loader, num_classes = preprocess_image_cnn(train_path, val_path, test_path)
    # Initialize the CNN model, loss function, and optimizer
    model = CNNClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer with learning rate 0.001
    # Train the CNN model using the training and validation data
    train_image_cnn(model, train_loader, val_loader, optimizer, criterion)
    # Evaluate the CNN model using the test data
    eval_image_cnn(model, test_loader)

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Entry point for the script: load data, preprocess, and execute various training/evaluation routines.
    """
    # csv_path = "kaggle/pokedex.csv"  # Path to main CSV data file
    # # Split the CSV data into training, validation, and test DataFrames
    # train_df, val_df, test_df = split_csv_data(csv_path)
    
    # # Train and evaluate the basic regression model
    # mse, r2 = basic_regression(train_df, test_df)
    # print(f"Mean Squared Error: {mse}")
    # print(f"R^2 Score: {r2}")
    
    # # Train and evaluate the optimized regression model
    # mse, r2 = optimized_regression(train_df, val_df, test_df)
    # print(f"Mean Squared Error: {mse}")
    # print(f"R^2 Score: {r2}")
    
    # Paths to image data for KNN and CNN models
    image_train_path = "kaggle/pokemon_by_type1/train/"
    image_val_path = "kaggle/pokemon_by_type1/val/"
    image_test_path = "kaggle/pokemon_by_type1/test/"
    
    # Train and evaluate KNN model on image data
    image_knn(3, image_train_path, image_val_path, image_test_path)
    
    # Train and evaluate CNN model on image data
    image_cnn(image_train_path, image_val_path, image_test_path)
    
    model = YOLO("runs/classify/train3/weights/best.pt")
    metrics = model.val("kaggle/pokemon_by_type1/test")
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category
