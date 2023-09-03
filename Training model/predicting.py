import os
import random
import keras
import numpy as np
from keras.preprocessing import image

# Load the pre-trained model
model = keras.models.load_model('Cats_vs_Dogs.model')

# Define the path to the test image folder
test_folder = r'archive\dogscats\test1'

# Get a list of all image filenames in the test folder
all_image_filenames = os.listdir(test_folder)

# Randomly select 10 image filenames for testing
random_image_filenames = random.sample(all_image_filenames, 10)

# Create a mapping to interpret class indices as 'Cat' or 'Dog'
class_mapping = {0: 'Cat', 1: 'Dog'}

# Loop through the randomly selected image filenames
for filename in random_image_filenames:
    # Load and resize the image to a consistent size (60x60 pixels)
    img = image.load_img(os.path.join(test_folder, filename), target_size=(60, 60, 1))
    
    # Convert the image to grayscale
    img = img.convert('L')
    
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    
    # Add an extra dimension to the array (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values to be in the range [0, 1]
    img_array /= 255.0

    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    
    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions)

    # Map the class index to its corresponding label ('Cat' or 'Dog')
    predicted_class = class_mapping[class_index]

    # Print the filename and the predicted class
    print(f"Filename: {filename}, Predicted Class: {predicted_class}")
