import numpy as np
import os
import cv2
import random
import pickle

# Define the directory where the dataset is located
DIRECTORY = r'archive\dogscats\train'

# Define the categories (classes) in the dataset
CATEGORIES = ['cats', 'dogs']

# Create an empty list to store image data and labels
data = []

# Loop through each category (class)
for category in CATEGORIES:
    # Create the path to the current category
    path = os.path.join(DIRECTORY, category)
    
    # Loop through each image file in the current category
    for img in os.listdir(path):
        # Create the full path to the image file
        img_path = os.path.join(path, img)
        
        # Assign a label to the current category (0 for 'cats', 1 for 'dogs')
        label = CATEGORIES.index(category)
        
        # Read the image in grayscale
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize the image to a consistent size (60x60 pixels)
        new_arr = cv2.resize(arr, (60, 60))
        
        # Append the resized image and its label to the data list
        data.append([new_arr, label])

# Shuffle the data to ensure randomness
random.shuffle(data)

# Separate the data into features (X) and labels (y)
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Convert the lists to NumPy arrays for compatibility with machine learning libraries
X = np.array(X)
y = np.array(y)

# Save the processed data as serialized files using pickle
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))
