import os
import shutil

import pandas as pd
from PIL import Image

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

    # Define the base directory for the new structure
    base_dir = "kaggle/pokemon_by_type1/train"

    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Iterate over each unique "Type 1" value
    for type1 in df['Type 1'].unique():
        # Create a subdirectory for each unique type
        type_dir = os.path.join(base_dir, type1)
        os.makedirs(type_dir, exist_ok=True)

        # Filter the DataFrame to get all rows with this "Type 1" value
        type_images = df[df['Type 1'] == type1]['Image']

        # Copy each image to the corresponding type directory
        for image_path in type_images:
            # Define the destination path
            dest_path = os.path.join(type_dir, os.path.basename(image_path))
            
            # Copy the image
            shutil.copy(f"kaggle/{image_path}", dest_path)

    print("Images have been organized by Type 1 in the 'pokemon_by_type1/train' directory.")
    
def image_size():
    # Get the size of the image
    image = Image.open("kaggle/pokemon_by_type1/train/Bug/14.png")
    print(image.size)
    
if __name__ == "__main__":
    types_list = main_type_count("kaggle/pokedex.csv")
    for type in types_list:
        print(f"{types_list.index(type)}:{type}")
    preprocess_classification("kaggle/pokedex.csv")
    image_size()