import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

import os
from flask import Flask, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')
model = keras.models.load_model("ds_logo3_maya_g.h5")
labels = {0: "bad", 1: "good"}

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    # If the user does not select a file, the browser should also
    # submit an empty part without filename
    if file.filename == '':
        return redirect(request.url)

    # Securely save the file
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Provide the uploaded image URL to the client
        uploaded_image_url = f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        
        image = Image.open(f"uploads\{filename}")
        image_array = np.array(image)
        resized_img = cv2.resize(image_array, (224,224))

        data = np.empty((1, 224, 224, 3)).astype(np.float32)
        data[0] = resized_img
        mb2_img = tf.keras.applications.mobilenet_v2.preprocess_input(data)

        prediction = np.argmax(model.predict(mb2_img), axis = 1)
        
        #print(f"uploaded_image_url = {uploaded_image_url}")
        return render_template('index.html', uploaded_image=uploaded_image_url, prediction=labels[prediction[0]])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()