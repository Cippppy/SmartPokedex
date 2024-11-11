import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def mnist_knn():
    # Load the MNIST dataset (or any other image dataset)
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target

    # Normalize pixel values (0-255 to 0-1)
    X = X / 255.0

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the KNN classifier with k neighbors
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the KNN classifier
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of KNN with k={k}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    mnist_knn()