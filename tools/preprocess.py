import os
import shutil

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

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
    
def image_size(image_path: str):
    # Get the size of the image
    image = Image.open(image_path)
    print(image.size)
    
if __name__ == "__main__":
    types_list = main_type_count("kaggle/pokedex.csv")
    for type in types_list:
        print(f"{types_list.index(type)}:{type}")
    preprocess_classification("kaggle/pokedex.csv")
    image_size("kaggle/pokemon_by_type1/test/Bug/18.png")