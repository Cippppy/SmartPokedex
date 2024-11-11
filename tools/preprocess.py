"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os
import shutil

# Third-party imports
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def main_type_count(csv_path: str):
    # Load the dataset
    df = pd.read_csv(csv_path)
    # Distribution of Pokémon types
    types = df['Type 1'].unique()
    types_list = types.tolist()
    return sorted(types_list)

def preprocess_classification(csv_path: str):
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
    # Load your data
    df = pd.read_csv(csv_path)

    # Split data into train (80%), val (10%), and test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df

# Function to load images and labels from a directory
def preprocess_image_knn(train_path: str, val_path: str, test_path: str):
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
                        # image.convert("L")  # Convert to grayscale
                        image = image.resize(image_size)  # Resize image
                        image_data = np.array(image).flatten()  # Flatten image to 1D
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
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images, adjust for RGB if needed
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
    # Get the size of the image
    image = Image.open(image_path)
    print(image.size)
 
# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #
    
if __name__ == "__main__":
    types_list = main_type_count("kaggle/pokedex.csv")
    for type in types_list:
        print(f"{types_list.index(type)}:{type}")
    preprocess_classification("kaggle/pokedex.csv")
    image_size("kaggle/pokemon_by_type1/test/Bug/18.png")