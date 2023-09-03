import os
import random
import keras
import numpy as np
from keras.preprocessing import image

model = keras.models.load_model('Cats_vs_Dogs.model')

test_folder = r'archive\dogscats\test1'

all_image_filenames = os.listdir(test_folder)

random_image_filenames = random.sample(all_image_filenames, 10)

class_mapping = {0: 'Cat', 1: 'Dog'}

for filename in random_image_filenames:
    img = image.load_img(os.path.join(test_folder, filename), target_size=(60, 60, 1))
    img = img.convert('L')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)

    predicted_class = class_mapping[class_index]

    print(f"Filename: {filename}, Predicted Class: {predicted_class}")
