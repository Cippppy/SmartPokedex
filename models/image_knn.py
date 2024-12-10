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
from tqdm import tqdm  # For progress bar

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def preprocess_image_knn(train_path: str, val_path: str, test_path: str, image_size=(120, 120), convert_to_grayscale=False):
    """
    Preprocess images for KNN model by resizing, flattening, and encoding labels.
    
    Args:
        train_path (str): Path to the training images directory.
        val_path (str): Path to the validation images directory.
        test_path (str): Path to the test images directory.
        image_size (tuple): Size to resize images to (width, height).
        convert_to_grayscale (bool): Whether to convert images to grayscale.
        
    Returns:
        tuple: Arrays of images and labels for train, validation, and test sets, along with the label encoder.
    """
    def load_images_from_directory(dataset_path):
        """
        Helper function to load and preprocess images from a directory.
        
        Args:
            dataset_path (str): Path to the dataset directory.
            
        Returns:
            tuple: Arrays of image data and corresponding labels.
        """
        images = []
        labels = []
        label_dirs = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]
        
        # Create a progress bar for loading all images
        with tqdm(total=sum(len(os.listdir(os.path.join(dataset_path, label))) 
                            for label in label_dirs), desc=f"Processing {dataset_path}") as pbar:
            for label in label_dirs:
                label_path = os.path.join(dataset_path, label)
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    try:
                        with Image.open(image_path) as image:
                            # Convert to grayscale or RGB based on the parameter
                            if convert_to_grayscale:
                                image = image.convert("L")
                            else:
                                image = image.convert("RGB")
                            # Resize and flatten the image
                            image = image.resize(image_size)
                            image_data = np.array(image).flatten()
                            images.append(image_data)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                    pbar.update(1)  # Update the progress bar
        return np.array(images), np.array(labels)
    
    print("Starting data preprocessing...")
    X_train, y_train = load_images_from_directory(train_path)
    X_val, y_val = load_images_from_directory(val_path)
    X_test, y_test = load_images_from_directory(test_path)
    print("Image loading complete.")

    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print("Label encoding complete.")
    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, label_encoder

def train_image_knn(X_train, y_train_encoded, X_val, y_val_encoded, k=3, sweep=False, patience=3):
    """
    Train a K-Nearest Neighbors (KNN) classifier for image classification.
    
    Args:
        X_train (array-like): Training data features.
        y_train_encoded (array-like): Encoded training data labels.
        X_val (array-like): Validation data features.
        y_val_encoded (array-like): Encoded validation data labels.
        k (int): Initial number of neighbors to use in the KNN model.
        sweep (bool): Whether to sweep through different \( k \)-values.
        patience (int): Number of iterations to allow no improvement before stopping the sweep.
        
    Returns:
        knn: Trained KNN model.
        best_k (int): Best \( k \)-value (or provided \( k \) if sweep=False).
        best_accuracy (float): Validation accuracy for the best \( k \)-value.
        ks (list): List of \( k \)-values tested (only if sweep=True).
        accuracys (list): Corresponding validation accuracies (only if sweep=True).
    """
    print("Initializing KNN model training...")
    if sweep:
        print("Starting k-value sweep...")
        ks = []
        accuracys = []
        best_k = k
        last_k = k
        current_k = k
        best_accuracy = -1

        while current_k > 0 and patience > 0:
            print(f"Testing k={current_k}...")
            knn = KNeighborsClassifier(n_neighbors=current_k)
            knn.fit(X_train, y_train_encoded)
            y_val_pred = knn.predict(X_val)
            val_accuracy = accuracy_score(y_val_encoded, y_val_pred)

            print(f"Validation accuracy with k={current_k}: {val_accuracy:.4f}")
            ks.append(current_k)
            accuracys.append(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_k = current_k
                patience = 3
                print(f"New best k={best_k} with accuracy={best_accuracy:.4f}. Resetting patience.")
                if last_k > current_k:
                    last_k = current_k
                    current_k -= 1
                else:
                    last_k = current_k
                    current_k += 1
            else:
                patience -= 1
                print(f"No improvement. Decreasing patience to {patience}.")
                if last_k > current_k:
                    last_k = current_k
                    current_k += 1
                else:
                    last_k = current_k
                    current_k -= 1

        print(f"Sweep complete. Best k={best_k} with accuracy={best_accuracy:.4f}.")
        return knn, best_k, best_accuracy, ks, accuracys

    print("Sweep disabled. Evaluating model with fixed k...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train_encoded)
    print(f"Model training completed with k={k}.")
    y_val_pred = knn.predict(X_val)
    val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
    print(f"Validation accuracy with fixed k={k}: {val_accuracy:.4f}")
    return knn, k, val_accuracy, [k], [val_accuracy]

def eval_image_knn(model, k, X_test, y_test_encoded, label_encoder):
    """
    Evaluate a K-Nearest Neighbors (KNN) classifier on test datasets.
    
    Args:
        model: Trained KNN model.
        k (int): Number of neighbors used in the KNN model.
        X_test (array-like): Test features.
        y_test_encoded (array-like): Encoded labels for the test set.
        label_encoder (LabelEncoder): Encoder to map numerical labels back to original labels.
    """
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    print(f"Test Accuracy of KNN with k={k}: {test_accuracy * 100:.2f}%")

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Parameters
    k = 1
    train_path = "data\merged_pokemon_by_type1/train"
    val_path = "data\merged_pokemon_by_type1/val"
    test_path = "data\merged_pokemon_by_type1/test"
    convert_to_grayscale = True
    sweep = True

    # Preprocess the images
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocess_image_knn(
        train_path, val_path, test_path, convert_to_grayscale=convert_to_grayscale
    )
    print("Data preprocessing complete.")

    # Train the KNN model
    model, best_k, _, _, _ = train_image_knn(
        X_train, y_train, X_val, y_val, k=k, sweep=sweep
    )
    print("KNN model training complete.")

    # Evaluate the model
    eval_image_knn(model, best_k, X_test, y_test, label_encoder)
    print("KNN model evaluation complete.")
    