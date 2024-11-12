"""
## Summary
This script provides functions for preprocessing Pokémon-related data, including organizing images 
by type for classification tasks, splitting datasets, and loading images with transformations for KNN and CNN models.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os
import shutil  # For file operations like copying images

# Third-party imports
import pandas as pd  # Data manipulation and analysis library
from PIL import Image  # Image processing library
import numpy as np  # Array manipulation
from sklearn.model_selection import train_test_split  # Dataset splitting
from sklearn.preprocessing import LabelEncoder  # Label encoding
from torch.utils.data import DataLoader  # Data loading for PyTorch
from torchvision import datasets, transforms  # Image transformations

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def main_type_count(csv_path: str):
    """
    Calculate and return a sorted list of unique Pokémon types in the dataset.
    
    Args:
        csv_path (str): Path to the CSV file containing Pokémon data.
        
    Returns:
        list: Sorted list of unique Pokémon types found in the "Type 1" column.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    # Extract unique Pokémon types from "Type 1" column
    types = df['Type 1'].unique()
    types_list = types.tolist()
    return sorted(types_list)

def preprocess_classification(csv_path: str):
    """
    Organize images by type into 'train', 'val', and 'test' folders for classification tasks.
    
    Args:
        csv_path (str): Path to the CSV file containing Pokémon data.
        
    Returns:
        None: Organizes images into folders within the "kaggle/pokemon_by_type1" directory.
    """
    # Load your data
    df = pd.read_csv(csv_path)

    # Define base directories for train, val, and test
    base_dir = "kaggle/pokemon_by_type1"
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)

    # Iterate over each unique "Type 1" value
    for type1 in df['Type 1'].unique():
        # Get all image paths for this type
        type_images = df[df['Type 1'] == type1]['Image'].tolist()

        # Split data into train (80%), val (10%), and test (10%)
        train_imgs, temp_imgs = train_test_split(type_images, test_size=0.2, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        # Define directories for each split and type
        for split, images in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_type_dir = os.path.join(base_dir, split, type1)
            os.makedirs(split_type_dir, exist_ok=True)

            # Copy each image to the corresponding split/type directory
            for image_path in images:
                dest_path = os.path.join(split_type_dir, os.path.basename(image_path))
                shutil.copy(f"kaggle/{image_path}", dest_path)

    print("Images have been organized by Type 1 into train, val, and test directories.")
    
def split_csv_data(csv_path: str):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        csv_path (str): Path to the CSV file containing data.
        
    Returns:
        tuple: DataFrames for train, validation, and test sets.
    """
    # Load your data
    df = pd.read_csv(csv_path)

    # Split data into train (80%), val (10%), and test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df

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
                        image = Image.open(image_path)
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

def preprocess_image_cnn(train_path: str, val_path: str, test_path: str):
    """
    Preprocess images for CNN model by applying transformations and loading data into PyTorch DataLoader.
    
    Args:
        train_path (str): Path to the training images directory.
        val_path (str): Path to the validation images directory.
        test_path (str): Path to the test images directory.
        
    Returns:
        tuple: DataLoader objects for train, validation, and test sets, along with the number of classes.
    """
    # Image transformations to resize, normalize, and convert to tensors
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjust normalization for RGB if necessary
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    
    return train_loader, val_loader, test_loader, num_classes

def image_size(image_path: str):
    """
    Print the size (width, height) of a specified image.
    
    Args:
        image_path (str): Path to the image file.
    """
    image = Image.open(image_path)
    print(image.size)

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #
    
if __name__ == "__main__":
    # Display the unique Pokémon types in the dataset
    types_list = main_type_count("kaggle/pokedex.csv")
    for type in types_list:
        print(f"{types_list.index(type)}:{type}")
    
    # Organize images into train, val, and test directories by Type 1
    preprocess_classification("kaggle/pokedex.csv")
    
    # Print the size of a sample image
    image_size("kaggle/pokemon_by_type1/test/Bug/18.png")
