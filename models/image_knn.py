"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os

# Third-party imports
from PIL import Image  # Image processing library
from sklearn.neighbors import KNeighborsClassifier  # KNN classifier
from sklearn.metrics import accuracy_score  # Evaluation metrics
import numpy as np  # Array manipulation
from sklearn.preprocessing import LabelEncoder  # Label encoding

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def preprocess_image_knn(train_path: str, val_path: str, test_path: str):
    """
    Preprocess images for KNN model by resizing and flattening them, then encoding labels.
    
    Args:
        train_path (str): Path to the training images directory.
        val_path (str): Path to the validation images directory.
        test_path (str): Path to the test images directory.
        
    Returns:
        tuple: Arrays of images and labels for train, validation, and test sets, along with the label encoder.
    """
    # Function to load images and labels from a directory
    def load_images_from_directory(dataset_path, image_size=(64, 64)):
        images = []
        labels = []
        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    try:
                        with Image.open(image_path) as image:
                            # Convert all images to RGB to ensure consistency
                            image = image.convert("RGB")
                            # Resize and flatten image data
                            image = image.resize(image_size)
                            image_data = np.array(image).flatten()  # Convert image to 1D array
                            images.append(image_data)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
        return np.array(images), np.array(labels)
    
    # Load images and labels from each split
    X_train, y_train = load_images_from_directory(train_path, image_size=(120, 120))
    X_val, y_val = load_images_from_directory(val_path, image_size=(120, 120))
    X_test, y_test = load_images_from_directory(test_path, image_size=(120, 120))

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, label_encoder

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

if __name__ == "__main__":
    
    # Training Parameters
    k = 1
    train_path = "data/combined_by_type1/train"
    val_path = "data/combined_by_type1/val"
    test_path = "data/combined_by_type1/test"
    
    # Preprocess the images for KNN
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocess_image_knn(train_path, val_path, test_path)
    print("Data preprocessing complete.")
    # Train the KNN model with the training data and specified k value
    model = train_image_knn(X_train, y_train, k)
    print("KNN model training complete.")
    # Evaluate the model using validation and test data
    eval_image_knn(model, k, X_val, y_val, X_test, y_test, label_encoder)
    print("KNN model evaluation complete.")