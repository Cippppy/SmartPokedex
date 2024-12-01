# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def preprocess_pokemon_data(csv_path):
    """
    Load Pokémon data from a CSV file and preprocess it for KNN classification.

    Args:
        csv_path (str): Path to the Pokémon CSV file.

    Returns:
        tuple: Features (X), encoded labels (y), and the label encoder.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract features and labels
    features = df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']].values
    labels = df['Type 1'].values

    # Encode the labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    return features, encoded_labels, label_encoder

def train_knn(X_train, y_train, k=3):
    """
    Train a K-Nearest Neighbors (KNN) classifier.

    Args:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        k (int): Number of neighbors to use in the KNN model.

    Returns:
        KNeighborsClassifier: Trained KNN model.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_knn(model, X, y, label_encoder, dataset_name="Dataset"):
    """
    Evaluate a KNN model and print the accuracy.

    Args:
        model: Trained KNN model.
        X (array-like): Features to evaluate.
        y (array-like): True labels.
        label_encoder (LabelEncoder): Encoder to map numerical labels back to original labels.
        dataset_name (str): Name of the dataset (e.g., "Train", "Test").
    """
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")
    
    # Example predictions (for debugging or insights)
    print(f"Example predictions from {dataset_name} set:")
    for i in range(5):
        print(f"True: {label_encoder.inverse_transform([y[i]])[0]}, Predicted: {label_encoder.inverse_transform([predictions[i]])[0]}")

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Path to the Pokémon dataset CSV file
    csv_path = "data/pokemon_w_stats_images/pokedex.csv"
    # csv_path = "data/combined_pokemon_dataset.csv"
    
    # Preprocess the data
    print("Preprocessing Pokémon data...")
    X, y, label_encoder = preprocess_pokemon_data(csv_path)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data preprocessing complete.")

    # Train the KNN model
    print("Training KNN model...")
    k = 2  # Number of neighbors
    knn_model = train_knn(X_train, y_train, k=k)
    print("KNN model training complete.")

    # Evaluate the model
    print("Evaluating KNN model...")
    evaluate_knn(knn_model, X_train, y_train, label_encoder, dataset_name="Train")
    evaluate_knn(knn_model, X_test, y_test, label_encoder, dataset_name="Test")
    print("KNN model evaluation complete.")
