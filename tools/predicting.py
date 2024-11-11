"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def eval_basic_regression(test_df: pd.DataFrame, model):
    X_test = test_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_test = test_df['Total']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def eval_image_knn(model, k, X_val, y_val_encoded, X_test, y_test_encoded, label_encoder):
    # Validate model on the validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation Accuracy of KNN with k={k}: {val_accuracy * 100:.2f}%")

    # Test the model on the test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    print(f"Test Accuracy of KNN with k={k}: {test_accuracy * 100:.2f}%")

    # To decode predicted labels back to original names
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
# Testing function
def eval_image_cnn(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    