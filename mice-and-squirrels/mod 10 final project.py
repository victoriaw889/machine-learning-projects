import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from skimage.util import random_noise
from collections import Counter

# Define the path to the dataset
dataset_path = r"C:\Users\vicky\OneDrive\Documents\Junior Year (11th Grade)\7th Period Machine Learning\mice and squirrels"

# Image parameters
image_size = (64, 64)  # Resize all images to 64x64 pixels

# Initialize lists for features and labels
X = []  # Features (image data)
y = []  # Labels (class names)

# Function to augment images
def augment_image(image):
    augmented_images = []
    # Original image
    augmented_images.append(image)
    # Add noise
    augmented_images.append(random_noise(image))
    # Rotate clockwise
    augmented_images.append(rotate(image, angle=45, mode='wrap'))
    # Rotate counterclockwise
    augmented_images.append(rotate(image, angle=-45, mode='wrap'))
    return augmented_images

# Loop through the folders and load images
for label in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, label)
    if os.path.isdir(class_folder):  # Check if it's a directory
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            try:
                # Load image
                image = imread(file_path, as_gray=True)  # Load as grayscale
                # Resize image to uniform size
                image_resized = resize(image, image_size, anti_aliasing=True)
                if label == "squirrels":  # Replace with the minority class label
                    augmented_images = augment_image(image_resized)
                    for img in augmented_images:
                        X.append(img.flatten())  # Flatten augmented images
                        y.append(label)         # Label for each augmented image
                else:
                    # Add non-augmented images for majority class
                    X.append(image_resized.flatten())
                    y.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# 1. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set distribution:", Counter(y_train))
print("Test set distribution:", Counter(y_test))

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the Model
y_pred = clf.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))