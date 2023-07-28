import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps,ImageTk

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gecmdvent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'keras_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

classes = {
            1:'Mild',
            2:'Moderate',
            3:'No DR',
            4:'Proliferative DR',
            5:'Severe'
           }

def model_predict(img_path, model):
    global prediction
    global final_prediction
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(img_path)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    idx = np.argmax(prediction)
    final_prediction = classes[idx+1]
    return final_prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    print("inside about")
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("inside post")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        percentage_list = []
        percentage_list.append(final_prediction)
        for i in range(len(prediction[0])):
            percentage_list.append("{0:.2%}".format(prediction[0][i]))
        print(percentage_list)
        return percentage_list
    return None


if __name__ == '__main__':
    app.run(debug=True)
