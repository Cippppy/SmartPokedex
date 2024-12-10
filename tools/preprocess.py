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
import random
import re

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
    
def organize_pokemon_images_by_type(csv_path, source_paths, destination_path):
    """
    Organizes Pokémon images into directories grouped by their primary type (Type1).
    
    Args:
        csv_path (str): Path to the CSV file containing Pokémon data.
        source_paths (list): List of source directories to search for Pokémon images.
        destination_path (str): Path to the destination directory where images will be organized.

    Returns:
        None
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract Type1 from the Type column (e.g., split by ',' and take the first type)
    df['Type1'] = df['Type'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else "Unknown")

    # Iterate over the DataFrame
    for _, row in df.iterrows():
        pokemon = row['Pokemon']
        type1 = row['Type1']
        
        # Create type1 directory in the destination path
        type1_path = os.path.join(destination_path, "train", type1)
        os.makedirs(type1_path, exist_ok=True)
        
        # Locate images for the current Pokémon in source folders
        for source_path in source_paths:
            pokemon_path = os.path.join(source_path, pokemon)
            if os.path.exists(pokemon_path):
                # Copy all files from the Pokémon directory to the new type1 directory
                for root, _, files in os.walk(pokemon_path):
                    for file in files:
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(type1_path, file)
                        shutil.copy(src_file, dest_file)

    print("Images have been grouped by Type1.")
    
def split_dataset_by_type1(images_root_path, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits images into train, val, and test sets for each Type 1, maintaining specified ratios.

    Args:
        images_root_path (str): Path where all images are currently stored (e.g., `train` folder).
        output_path (str): Path to the root directory for the split dataset.
        train_ratio (float): Proportion of images to include in the training set.
        val_ratio (float): Proportion of images to include in the validation set.
        test_ratio (float): Proportion of images to include in the test set.

    Returns:
        None
    """
    # Create output directories
    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")

    for dir_path in [train_path, val_path, test_path]:
        os.makedirs(dir_path, exist_ok=True)

    # Process each Type 1
    for type1_dir in os.listdir(images_root_path):
        type1_path = os.path.join(images_root_path, type1_dir)
        if not os.path.isdir(type1_path):
            continue

        # Collect all images for the current Type 1
        images = [file for file in os.listdir(type1_path) if os.path.isfile(os.path.join(type1_path, file))]
        random.shuffle(images)

        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)

        # Split the images
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        # Helper function to copy files
        def copy_files(image_list, destination_folder):
            dest_dir = os.path.join(destination_folder, type1_dir)
            os.makedirs(dest_dir, exist_ok=True)
            for image in image_list:
                shutil.copy(os.path.join(type1_path, image), os.path.join(dest_dir, image))

        # Copy files to respective folders
        copy_files(train_images, train_path)
        copy_files(val_images, val_path)
        copy_files(test_images, test_path)

    print("Dataset split complete.")
    print(f"Train set: {train_path}")
    print(f"Validation set: {val_path}")
    print(f"Test set: {test_path}")
    
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
    
def combine_csv_data(csv_paths: list, output_path: str):
    """
    Combine multiple CSV files into a single CSV file, handling type columns properly.
    
    Args:
        csv_paths (list): List of paths to CSV files to combine.
        output_path (str): Path to the output CSV file.
    """
    # Load datasets
    df0 = pd.read_csv(csv_paths[0])  # First CSV file
    df1 = pd.read_csv(csv_paths[1])  # Second CSV file

    # Standardize column names for merging
    df0.rename(columns={"Name": "Pokemon"}, inplace=True)

    # Adjust column names in the first dataset for compatibility
    stat_mapping = {
        "HP": "HP Base",
        "Attack": "Attack Base",
        "Defense": "Defense Base",
        "Speed": "Speed Base",
        "SP. Atk.": "Special Attack Base",
        "SP. Def": "Special Defense Base",
        "Type 1": "Type 1",
        "Type 2": "Type 2"
    }
    df0.rename(columns=stat_mapping, inplace=True)

    # Split the "Type" column in df1 into "Type 1" and "Type 2"
    if "Type" in df1.columns:
        df1[["Type 1", "Type 2"]] = df1["Type"].str.split(",", expand=True).fillna("").apply(lambda x: x.str.strip())
        df1.drop(columns=["Type"], inplace=True)  # Drop the original "Type" column

    # Merge datasets on the Pokémon name (key)
    merged_df = pd.merge(df0, df1, on="Pokemon", how="outer")

    # Combine `_x` and `_y` columns where applicable
    for stat in stat_mapping.values():  # Iterate over the 'Base' stat columns
        if f"{stat}_x" in merged_df.columns and f"{stat}_y" in merged_df.columns:
            # Prioritize `_x` if it exists; fallback to `_y`
            merged_df[stat] = merged_df[f"{stat}_x"].combine_first(merged_df[f"{stat}_y"])
            # Drop the original `_x` and `_y` columns
            merged_df.drop(columns=[f"{stat}_x", f"{stat}_y"], inplace=True)

    # Save the merged dataset to a new CSV
    merged_df.to_csv(output_path, index=False)
    
    print(f"Combined data saved to: {output_path}")
    
def organize_images(stats_csv_path, src1_path, src2_path, src3_path, dest_path):
    """
    Organizes images from multiple sources into a single directory structure organized by 'Type 1'.

    Args:
        stats_csv_path (str): Path to the CSV containing Pokémon stats (with 'Image', 'Index', and 'Type 1' columns).
        src1_path (str): Path to the first source directory (images labeled '1.png', '2.png', etc.).
        src2_path (str): Path to the second source directory (subdirectories for each Pokémon with more subdirectories).
        src3_path (str): Path to the third source directory (subdirectories for each Pokémon with images).
        dest_path (str): Path to the destination directory (organized by 'Type 1').

    Returns:
        None
    """
    # Load the stats CSV to map Pokémon indices and images to 'Type 1'
    df = pd.read_csv(stats_csv_path)
    # Strip whitespace and ensure consistency in the 'Type 1' column
    df['Type 1'] = df['Type 1'].str.strip()
    df['Pokemon'] = df['Pokemon'].str.strip()
    df['Image'] = df['Image'].str.strip()
    df['Index'] = df['Index'].fillna(0).astype(int)  # Handle NaN in 'Index'

    # Map keys to Type 1
    image_to_type1 = dict(zip(df['Image'], df['Type 1']))
    index_to_type1 = dict(zip(df['Index'], df['Type 1']))
    name_to_type1 = dict(zip(df['Pokemon'], df['Type 1']))
    
    # Ensure the destination directory exists
    os.makedirs(dest_path, exist_ok=True)

    # Helper function to copy an image to the correct 'Type 1' directory
    def copy_to_type1(image_path, pokemon_identifier):
        # Determine the Type 1 for the given identifier (index, image name, or Pokémon name)
        type1 = index_to_type1.get(pokemon_identifier) or \
                image_to_type1.get(image_path) or \
                name_to_type1.get(pokemon_identifier)
        if not type1:
            print(f"Type 1 not found for {pokemon_identifier}, skipping...")
            return
        type1_dir = os.path.join(dest_path, type1)
        os.makedirs(type1_dir, exist_ok=True)
        shutil.copy(image_path, type1_dir)

    # Process images in the first dataset (e.g., '1.png', '2.png', etc.)
    for image_file in os.listdir(src1_path):
        if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
            index = os.path.splitext(image_file)[0]  # Get index from filename
            image_path = os.path.join(src1_path, image_file)
            copy_to_type1(image_path, int(index))

    # Process images in the second dataset (subdirectories with Pokémon names)
    for pokemon_dir in os.listdir(src2_path):
        pokemon_path = os.path.join(src2_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            for sub_dir in os.listdir(pokemon_path):  # Process subdirectories
                sub_path = os.path.join(pokemon_path, sub_dir)
                if os.path.isdir(sub_path):
                    for image_file in os.listdir(sub_path):
                        if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
                            image_path = os.path.join(sub_path, image_file)
                            copy_to_type1(image_path, pokemon_dir)

    # Process images in the third dataset (subdirectories with Pokémon names)
    for pokemon_dir in os.listdir(src3_path):
        pokemon_path = os.path.join(src3_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            for image_file in os.listdir(pokemon_path):
                if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
                    image_path = os.path.join(pokemon_path, image_file)
                    copy_to_type1(image_path, pokemon_dir)

    print(f"Images successfully organized into {dest_path}")

def organize_images_without_index(stats_csv_path, src1_path, src2_path, src3_path, dest_path):
    """
    Organizes images from multiple sources into a single directory structure organized by 'Type 1',
    without relying on the 'Index' column.
    
    Args:
        stats_csv_path (str): Path to the CSV containing Pokémon stats (with 'Image', 'Pokemon', and 'Type 1' columns).
        src1_path (str): Path to the first source directory (images labeled '1.png', '2.png', etc.).
        src2_path (str): Path to the second source directory (subdirectories for each Pokémon with more subdirectories).
        src3_path (str): Path to the third source directory (subdirectories for each Pokémon with images).
        dest_path (str): Path to the destination directory (organized by 'Type 1').

    Returns:
        None
    """
    # Load the stats CSV
    df = pd.read_csv(stats_csv_path)

    # Clean up columns to ensure consistent matching
    df['Type 1'] = df['Type 1'].str.strip()
    df['Pokemon'] = df['Pokemon'].str.strip()
    # Normalize the 'Image' column to match actual filenames
    df['Image'] = df['Image'].str.replace('images/', '').str.strip()

    # Create a mapping for `Image` to `Type 1` and `Pokemon` to `Type 1`
    image_to_type1 = dict(zip(df['Image'], df['Type 1']))
    name_to_type1 = dict(zip(df['Pokemon'], df['Type 1']))

    # Ensure the destination directory exists
    os.makedirs(dest_path, exist_ok=True)

    # Helper function to copy an image to the correct `Type 1` directory
    def copy_to_type1(image_path, identifier):
        # Determine the `Type 1` based on `Image` or `Pokemon` name
        type1 = image_to_type1.get(os.path.basename(image_path)) or name_to_type1.get(identifier)
        if not type1:
            print(f"Type 1 not found for {identifier}, skipping...")
            return
        type1_dir = os.path.join(dest_path, type1)
        os.makedirs(type1_dir, exist_ok=True)
        shutil.copy(image_path, type1_dir)

    # Process images in the first dataset (e.g., '1.png', '2.png', etc.)
    for image_file in os.listdir(src1_path):
        if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
            image_path = os.path.join(src1_path, image_file)
            copy_to_type1(image_path, os.path.basename(image_file))

    # Process images in the second dataset (subdirectories with Pokémon names)
    for pokemon_dir in os.listdir(src2_path):
        pokemon_path = os.path.join(src2_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            for sub_dir in os.listdir(pokemon_path):  # Process subdirectories
                sub_path = os.path.join(pokemon_path, sub_dir)
                if os.path.isdir(sub_path):
                    for image_file in os.listdir(sub_path):
                        if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
                            image_path = os.path.join(sub_path, image_file)
                            copy_to_type1(image_path, pokemon_dir)

    # Process images in the third dataset (subdirectories with Pokémon names)
    for pokemon_dir in os.listdir(src3_path):
        pokemon_path = os.path.join(src3_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            for image_file in os.listdir(pokemon_path):
                if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".jpeg") or image_file.endswith(".gif"):
                    image_path = os.path.join(pokemon_path, image_file)
                    copy_to_type1(image_path, pokemon_dir)

    print(f"Images successfully organized into {dest_path}")
    
def organize_images_handle_forms(stats_csv_path, src1_path, src2_path, src3_path, dest_path):
    """
    Organizes images from multiple sources into a single directory structure organized by 'Type 1',
    handling Pokémon forms by normalizing names to account for directory and CSV discrepancies.
    """
    # Load the stats CSV
    df = pd.read_csv(stats_csv_path)

    # Normalize the 'Pokemon' column
    def normalize_pokemon_name(name):
        """
        Normalize Pokémon names by handling discrepancies in form naming conventions.
        For example:
        - Aegislash (Blade Forme) -> Aegislash Blade Forme
        - Aegislash Blade Form -> Aegislash Blade Forme
        """
        # Remove parentheses and extra spaces
        name = re.sub(r"\((.*?)\)", r"\1", name).strip()
        # Standardize "Form" vs "Forme"
        name = name.replace(" Form ", " Forme ").replace(" Forme", " Forme")
        return name.strip()

    # Clean up columns to ensure consistent matching
    df['Type 1'] = df['Type 1'].str.strip()
    df['Pokemon_Normalized'] = df['Pokemon'].apply(normalize_pokemon_name)  # Add normalized names
    df['Image'] = df['Image'].str.replace('images/', '').str.strip()

    # Create mappings for normalized and original Pokémon names
    image_to_type1 = dict(zip(df['Image'], df['Type 1']))
    name_to_type1 = dict(zip(df['Pokemon'], df['Type 1']))
    normalized_name_to_type1 = dict(zip(df['Pokemon_Normalized'], df['Type 1']))

    # Ensure the destination directory exists
    os.makedirs(dest_path, exist_ok=True)

    # Helper function to copy an image to the correct `Type 1` directory
    def copy_to_type1(image_path, identifier, is_normalized=False):
        """
        Determine the `Type 1` directory for a Pokémon based on its identifier and copy the image.
        """
        if is_normalized:
            type1 = normalized_name_to_type1.get(identifier)
        else:
            type1 = name_to_type1.get(identifier) or normalized_name_to_type1.get(identifier)
        if not type1:
            print(f"Type 1 not found for {identifier} (image: {image_path}), skipping...")
            return
        print(f"Matched {identifier} (image: {image_path}) to Type 1: {type1}")
        type1_dir = os.path.join(dest_path, type1)
        os.makedirs(type1_dir, exist_ok=True)
        shutil.copy(image_path, type1_dir)

    # Process images in the first dataset (e.g., '1.png', '2.png', etc.)
    for image_file in os.listdir(src1_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(src1_path, image_file)
            copy_to_type1(image_path, os.path.basename(image_file))

    # Process images in the second dataset (subdirectories with Pokémon names)
    for pokemon_dir in os.listdir(src2_path):
        pokemon_path = os.path.join(src2_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            normalized_name = normalize_pokemon_name(pokemon_dir)
            for sub_dir in os.listdir(pokemon_path):  # Process subdirectories
                sub_path = os.path.join(pokemon_path, sub_dir)
                if os.path.isdir(sub_path):
                    for image_file in os.listdir(sub_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            image_path = os.path.join(sub_path, image_file)
                            copy_to_type1(image_path, pokemon_dir, is_normalized=False)
                            copy_to_type1(image_path, normalized_name, is_normalized=True)

    # Process images in the third dataset (subdirectories with Pokémon forms)
    for pokemon_dir in os.listdir(src3_path):
        pokemon_path = os.path.join(src3_path, pokemon_dir)
        if os.path.isdir(pokemon_path):  # Only process directories
            normalized_name = normalize_pokemon_name(pokemon_dir)
            for image_file in os.listdir(pokemon_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(pokemon_path, image_file)
                    copy_to_type1(image_path, pokemon_dir, is_normalized=False)
                    copy_to_type1(image_path, normalized_name, is_normalized=True)

    print(f"Images successfully organized into {dest_path}")
    
def split_dataset(source_dir, dest_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Splits a dataset into train, test, and validation sets.

    Args:
        source_dir (str): Path to the source directory containing images organized by type.
        dest_dir (str): Path to the destination directory for split datasets.
        train_ratio (float): Proportion of images for the training set.
        test_ratio (float): Proportion of images for the test set.
        val_ratio (float): Proportion of images for the validation set.

    Returns:
        None
    """
    # Ensure the destination directories exist
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    val_dir = os.path.join(dest_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Walk through each type folder in the source directory
    for type_folder in os.listdir(source_dir):
        type_path = os.path.join(source_dir, type_folder)
        if not os.path.isdir(type_path):
            continue

        # Get all files in the current type directory
        files = [f for f in os.listdir(type_path) if os.path.isfile(os.path.join(type_path, f))]
        random.shuffle(files)  # Shuffle files randomly

        # Calculate split sizes
        total_files = len(files)
        train_count = int(total_files * train_ratio)
        test_count = int(total_files * test_ratio)
        val_count = total_files - train_count - test_count

        # Split the files
        train_files = files[:train_count]
        test_files = files[train_count:train_count + test_count]
        val_files = files[train_count + test_count:]

        # Define destination paths for the type
        train_type_dir = os.path.join(train_dir, type_folder)
        test_type_dir = os.path.join(test_dir, type_folder)
        val_type_dir = os.path.join(val_dir, type_folder)

        os.makedirs(train_type_dir, exist_ok=True)
        os.makedirs(test_type_dir, exist_ok=True)
        os.makedirs(val_type_dir, exist_ok=True)

        # Move files to the respective directories
        for file in train_files:
            shutil.copy(os.path.join(type_path, file), os.path.join(train_type_dir, file))
        for file in test_files:
            shutil.copy(os.path.join(type_path, file), os.path.join(test_type_dir, file))
        for file in val_files:
            shutil.copy(os.path.join(type_path, file), os.path.join(val_type_dir, file))

        print(f"Split for {type_folder}: Train={len(train_files)}, Test={len(test_files)}, Val={len(val_files)}")

    print("Dataset successfully split into train, test, and val sets.")

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #
    
if __name__ == "__main__":
    # # Display the unique Pokémon types in the dataset
    # types_list = main_type_count("kaggle/pokedex.csv")
    # for type in types_list:
    #     print(f"{types_list.index(type)}:{type}")
    
    # # Organize images into train, val, and test directories by Type 1
    # preprocess_classification("kaggle/pokedex.csv")
    
    # # Print the size of a sample image
    # image_size("kaggle/pokemon_by_type1/test/Bug/18.png")
    
    # organize_pokemon_images_by_type("data\pokemon_img_data\pokemonDB_dataset.csv", 
    #                                 ["data\pokemon_img_data\Pokemon Dataset", "data\pokemon_img_data\Pokemon Images DB"], 
    #                                 "data/pokemon_img_data/pokemon_by_type1")
    # images_root_path = "data/combined_by_type1/all"
    # output_path = "data/combined_by_type1"
    # split_dataset_by_type1(images_root_path, output_path)
    # combine_csv_data(["data/pokemon_w_stats_images/pokedex.csv",
    #                   "data/pokemon_img_data/pokemonDB_dataset.csv"], 
    #                  "data/combined_dataset.csv")
    # organize_images_handle_forms("data/combined_dataset.csv",
    #                 "data/pokemon_w_stats_images/images",
    #                 "data/pokemon_img_data/Pokemon Dataset",
    #                 "data/pokemon_img_data/Pokemon Images DB",
    #                 "data/merged_imgs/all")
    split_dataset("data\merged_pokemon_by_type1/all",
                  "data\merged_pokemon_by_type1")
