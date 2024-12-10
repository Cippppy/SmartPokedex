import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def combine_pokemon_datasets(csv1_path, csv2_path, output_path):
    """
    Combines two Pokémon datasets into one, keeping Type 1 and Type 2 as separate columns 
    and eliminating redundant columns such as 'Image'.
    
    Args:
        csv1_path (str): Path to the first CSV file.
        csv2_path (str): Path to the second CSV file.
        output_path (str): Path to save the combined CSV file.
    
    Returns:
        None
    """
    # Load both datasets
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Standardize column names in df2 to align with df1
    df2 = df2.rename(columns={
        "Type 1": "Type1",
        "Type 2": "Type2",
        "HP": "HP Base",
        "Attack": "Attack Base",
        "Defense": "Defense Base",
        "SP. Atk.": "Special Attack Base",
        "SP. Def": "Special Defense Base",
        "Speed": "Speed Base"
    })

    # Drop the Image column from df2
    if "Image" in df2.columns:
        df2 = df2.drop(columns=["Image"])

    # Standardize Type1 and Type2 columns in df1
    df1[['Type1', 'Type2']] = df1['Type'].str.split(',', expand=True).apply(lambda x: x.str.strip())
    df1 = df1.drop(columns=["Type"], errors='ignore')  # Remove old Type column

    # Align columns between the two datasets
    combined_columns = set(df1.columns).union(df2.columns)
    df1 = df1.reindex(columns=combined_columns)
    df2 = df2.reindex(columns=combined_columns)

    # Combine the datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save the combined dataset to a new CSV file
    combined_df.to_csv(output_path, index=False)

    print(f"Combined dataset saved to {output_path}")

def analyze_pokemon_csv(csv_path, output_txt_path, output_plot_path):
    """
    Analyzes a Pokémon dataset, printing and saving statistics to a text file
    and creating a plot of Pokémon counts by Type 1.
    """
    df = pd.read_csv(csv_path)
    total_pokemon = len(df)
    type1_counts = df['Type 1'].value_counts()

    output_lines = [
        "#### Analysis of Pokémon Dataset ####\n",
        f"Total number of Pokémon: {total_pokemon}\n\n",
        "Number of Pokémon by Type 1:\n",
    ]
    output_lines.extend([f"  {type1}: {count}" for type1, count in type1_counts.items()])
    
    with open(output_txt_path, "w") as txt_file:
        txt_file.writelines(line + "\n" for line in output_lines)
    
    print("\n".join(output_lines))
    print(f"\nAnalysis output saved to {output_txt_path}")
    print(f"Plot saved to {output_plot_path}")
    plt.figure(figsize=(10, 6))
    type1_counts.plot(kind='bar', color='skyblue')
    plt.title("Number of Pokémon by Type 1")
    plt.xlabel("Type 1")
    plt.ylabel("Number of Pokémon")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot_path)

def count_images_by_type1(images_root_path, output_txt_path):
    """
    Counts the number of images per Type 1 in a given directory structure.
    """
    type1_image_counts = {}
    for type1_dir in os.listdir(images_root_path):
        type1_path = os.path.join(images_root_path, type1_dir)
        if os.path.isdir(type1_path):
            image_count = len([
                file for file in os.listdir(type1_path)
                if os.path.isfile(os.path.join(type1_path, file))
            ])
            type1_image_counts[type1_dir] = image_count

    total_images = sum(type1_image_counts.values())
    output_lines = [
        "#### Image Count by Type 1 ####\n",
        f"Total number of images: {total_images}\n\n",
        "Number of images by Type 1:\n"
    ]
    output_lines.extend([f"  {type1}: {count}" for type1, count in type1_image_counts.items()])

    with open(output_txt_path, "w") as txt_file:
        txt_file.writelines(line + "\n" for line in output_lines)

    print("\n".join(output_lines))
    print(f"\nResults saved to {output_txt_path}")

def analyze_image_sizes(image_directory):
    """
    Analyzes the sizes of images in a directory and its subdirectories.
    """
    image_sizes = []
    total_images = 0

    for root, _, files in os.walk(image_directory):
        for file in files:
            image_path = os.path.join(root, file)
            try:
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
                    total_images += 1
            except Exception:
                pass

    if not image_sizes:
        return {"error": "No valid images found"}

    largest_size = max(image_sizes, key=lambda x: x[0] * x[1])
    smallest_size = min(image_sizes, key=lambda x: x[0] * x[1])
    average_width = sum(size[0] for size in image_sizes) / total_images
    average_height = sum(size[1] for size in image_sizes) / total_images
    unique_sizes = set(image_sizes)
    size_counts = Counter(image_sizes)

    results = {
        "total_images": total_images,
        "largest_size": largest_size,
        "smallest_size": smallest_size,
        "average_size": (round(average_width, 2), round(average_height, 2)),
        "unique_sizes": list(unique_sizes),
        "size_counts": dict(size_counts),
    }

    print("\n#### Image Size Analysis ####")
    print(f"Total Images: {results['total_images']}")
    print(f"Largest Size: {results['largest_size']}")
    print(f"Smallest Size: {results['smallest_size']}")
    print(f"Average Size: {results['average_size']}")
    print(f"Unique Sizes: {len(results['unique_sizes'])} unique sizes")
    print("\nCounts of Each Size:")
    for size, count in results['size_counts'].items():
        print(f"  {size}: {count} images")

    return results

def count_images_in_directory(directory_path):
    """
    Counts the number of image files in a directory and its subdirectories.
    
    Args:
        directory_path (str): Path to the directory.
        
    Returns:
        int: Total number of image files found.
    """
    # Define image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    total_images = 0

    # Walk through the directory and count files with valid extensions
    for root, _, files in os.walk(directory_path):
        total_images += sum(1 for file in files if file.lower().endswith(tuple(image_extensions)))

    return total_images

if __name__ == "__main__":
    # csv_path = "data/merged_dataset.csv"
    # output_txt_path = "outputs/pokemon_analysis.txt"
    # output_plot_path = "outputs/pokemon_analysis_plot.png"
    # analyze_pokemon_csv(csv_path, output_txt_path, output_plot_path)
    
    # images_root_path = "data/combined_by_type1/all"
    # output_txt_path = "outputs/all_counts_by_type1.txt"
    # count_images_by_type1(images_root_path, output_txt_path)
    
    # images_root_path = "data\merged_pokemon_by_type1\\all"
    # print(count_images_in_directory(images_root_path))
    
    # images_root_path = "data/merged_imgs/all"
    # print(count_images_in_directory(images_root_path))
    
    # images_root_path = "data/pokemon_img_data/pokemon_by_type1/train"
    # print(count_images_in_directory(images_root_path))
    
    image_directory = "data\merged_pokemon_by_type1\\all"
    analysis_results = analyze_image_sizes(image_directory)
    # Print analysis results
    print(f"Total Images: {analysis_results['total_images']}")
    print(f"Largest Size: {analysis_results['largest_size']}")
    print(f"Smallest Size: {analysis_results['smallest_size']}")
    print(f"Average Size: {analysis_results['average_size']}")
    print(f"Unique Sizes: {len(analysis_results['unique_sizes'])} unique sizes")
    # print("Counts of Each Size:")
    # for size, count in analysis_results['size_counts'].items():
    #     print(f"  {size}: {count} images")