"""
Smart Pokedex

## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Local imports
from tools.models import CNNClassifier
from tools.preprocess import split_csv_data, preprocess_image_knn, preprocess_image_cnn
from tools.training import train_basic_regression, train_yolo, train_optimized_regression, train_image_knn, train_image_cnn
from tools.predicting import eval_basic_regression, eval_image_knn, eval_image_cnn

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def basic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame):
    model = train_basic_regression(train_df)
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

def optimized_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    model = train_optimized_regression(train_df, val_df)
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

def image_knn(k, train_path, val_path, test_path):
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocess_image_knn(train_path, val_path, test_path)
    model = train_image_knn(X_train, y_train, k)
    eval_image_knn(model, k, X_val, y_val, X_test, y_test, label_encoder)
    
def image_cnn(train_path, val_path, test_path, device='cuda'):
    train_loader, val_loader, test_loader, num_classes = preprocess_image_cnn(train_path, val_path, test_path)
    # Initialize model, loss function, and optimizer
    model = CNNClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_image_cnn(model, train_loader, val_loader, optimizer, criterion)
    eval_image_cnn(model, test_loader)
    
# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    csv_path = "kaggle/pokedex.csv"
    train_df, val_df, test_df = split_csv_data(csv_path)
    mse, r2 = basic_regression(train_df, test_df)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    
    mse, r2 = optimized_regression(train_df, val_df, test_df)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    
    image_train_path = "kaggle/pokemon_by_type1/train/"
    image_val_path = "kaggle/pokemon_by_type1/val/"
    image_test_path = "kaggle/pokemon_by_type1/test/"
    # train_yolo("yolo11n-cls.pt")
    image_knn(3, image_train_path, image_val_path, image_test_path)
    image_cnn(image_train_path, image_val_path, image_test_path)
    
    
    