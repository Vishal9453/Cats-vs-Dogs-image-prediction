from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = keras.models.load_model('Cats_vs_Dogs.model')

@app.route('/')
def index():
    # Serve an HTML page
    with open('index.html', 'r') as file:
        html_content = file.read()
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    imagefile = request.files['imagefile']

    # Read the image file as a binary stream
    img_stream = imagefile.read()

    # Load and preprocess the image
    img = image.load_img(io.BytesIO(img_stream), color_mode='grayscale', target_size=(60, 60))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction using the loaded model
    prediction = model.predict(img_array)
    
    # Determine the predicted class (Cat or Dog)
    predicted_class = "Dog" if prediction[0][1] > prediction[0][0] else "Cat"

    # Return the prediction as JSON
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run() # Start the Flask app
