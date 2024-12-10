"""
## Summary

"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Third-party imports
from ultralytics import YOLO  # YOLO model for object detection

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def train_yolo(model_path: str, data_path: str):
    """
    Train a YOLO model on Pokémon image data for classification.
    
    Args:
        model_path (str): Path to the pretrained YOLO model or model configuration file.
        
    Returns:
        results: Training results from the YOLO model.
    """
    # Load the YOLO model
    model = YOLO(model_path)  # Load a pretrained YOLO model or model yaml file

    # Train the model on the Pokémon image dataset with specified settings
    results = model.train(data=data_path, 
                          epochs=20, 
                          imgsz=640, 
                          device='cuda', 
                          batch=8,
                          val=True)

    return results

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    data_path = "data/merged_pokemon_by_type1"  # Path to the Pokémon image dataset
    # Run YOLO training as a standalone example
    # train_yolo("models/yolo11n-cls.pt", data_path)
    # train_yolo("models/yolov8n-cls.pt", data_path)
    train_yolo("cfgs/models/yolo11-poke.yaml", data_path)