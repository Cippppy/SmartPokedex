"""
## Summary
This script contains functions for evaluating trained models, including regression models,
K-Nearest Neighbors (KNN), and Convolutional Neural Networks (CNN).
Each function calculates relevant metrics to assess model performance on test data.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch  # PyTorch library for tensor operations and deep learning
import pandas as pd  # Data manipulation and analysis library
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score  # Evaluation metrics

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def eval_basic_regression(test_df: pd.DataFrame, model):
    """
    Evaluate a regression model on test data using Mean Squared Error (MSE) and R-squared (R2) metrics.
    
    Args:
        test_df (pd.DataFrame): DataFrame containing test data with features and target.
        model: Trained regression model to evaluate.
        
    Returns:
        tuple: Mean Squared Error (mse) and R^2 score (r2) of the model on the test data.
    """
    # Extract features and target from test dataset
    X_test = test_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_test = test_df['Total']
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate model using MSE and R2 metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def eval_image_knn(model, k, X_val, y_val_encoded, X_test, y_test_encoded, label_encoder):
    """
    Evaluate a K-Nearest Neighbors (KNN) classifier on validation and test datasets.
    
    Args:
        model: Trained KNN model.
        k (int): Number of neighbors used in the KNN model.
        X_val (array-like): Validation features.
        y_val_encoded (array-like): Encoded labels for validation set.
        X_test (array-like): Test features.
        y_test_encoded (array-like): Encoded labels for test set.
        label_encoder (LabelEncoder): Encoder to map numerical labels back to original labels.
    """
    # Predict and calculate accuracy on the validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation Accuracy of KNN with k={k}: {val_accuracy * 100:.2f}%")

    # Predict and calculate accuracy on the test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    print(f"Test Accuracy of KNN with k={k}: {test_accuracy * 100:.2f}%")

    # Decode predicted labels back to their original names if needed
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
def eval_image_cnn(model, test_loader, device='cuda'):
    """
    Evaluate a Convolutional Neural Network (CNN) on test data, calculating test accuracy.
    
    Args:
        model: Trained CNN model.
        test_loader: DataLoader for the test data.
        device (str, optional): Device to perform evaluation on ('cuda' for GPU, 'cpu' for CPU).
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient calculation for efficiency during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get class predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
