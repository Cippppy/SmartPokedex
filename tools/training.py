"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
import torch
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def train_basic_regression(train_df: pd.DataFrame):
    X_train = train_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_train = train_df['Total']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def train_optimized_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, model_type="ridge"):
    # Extract features and target from the training dataset
    X_train = train_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_train = train_df['Total']
    
    # Extract features and target from the validation dataset
    X_val = val_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_val = val_df['Total']

    # Choose model and set up hyperparameter grid
    if model_type == "ridge":
        model = Ridge()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == "lasso":
        model = Lasso()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    else:
        raise ValueError("model_type must be either 'ridge' or 'lasso'")

    # Perform grid search with cross-validation on training data
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
    # Load a model
    model = YOLO(model_path)  # load a pretrained model or model yaml file

    # Train the model
    results = model.train(data="kaggle/pokemon_by_type1/", epochs=20, imgsz=128, device='cuda')

    return results

# Training function
def train_image_cnn(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cuda'):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
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
    # Initialize the KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN classifier
    knn.fit(X_train, y_train)

    return knn

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    train_yolo("yolo11n-cls.pt")