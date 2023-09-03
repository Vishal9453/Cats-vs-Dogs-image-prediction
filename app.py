from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import keras
import numpy as np
from keras.preprocessing import image
import io

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

model = keras.models.load_model('Cats_vs_Dogs.model')

@app.route('/')
def index():
    with open('index.html', 'r') as file:
        html_content = file.read()
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']

    img_stream = imagefile.read()

    img = image.load_img(io.BytesIO(img_stream), color_mode='grayscale', target_size=(60, 60))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = "Dog" if prediction[0][1] > prediction[0][0] else "Cat"

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
