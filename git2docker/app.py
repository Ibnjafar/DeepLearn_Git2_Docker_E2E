import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
import json

# load the keras model
MODEL_PATH = '/root/app/git2docker/model.h5'


# Load the model from saved file
classifier = keras.models.load_model(MODEL_PATH)

# create the main application
app = Flask(__name__, static_url_path='')

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(tf.__version__)

# Make prediction for image and populate result dictionary
def make_prediction(filepath):
    result = {}
    result['filename'] = filepath
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    test_image = image.load_img(filepath, grayscale=True , target_size = (28,28))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1,28,28,1)
    result['answer'] = classifier.predict_classes(test_image)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    # save image to folder on disk
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    # Make the Prediction with saved image
    result = make_prediction(file.filename)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=24000, debug=True)
