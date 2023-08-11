# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:52:36 2023

@author: vinod
"""
from keras.models import load_model
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
cnn = Flask(__name__)

model = load_model(r"D:\College Files\Flask Codes\mnist\CNNModel.h5")
input_directory = (r"D:\College Files\Flask Codes\mnist")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    image_array = img_to_array(image)
    image_array /= 255.0
    image_array = image_array.reshape((1,) + image_array.shape)
    return image_array
    

@cnn.route('/')
def main():
    return render_template('get_image.html')

@cnn.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    f.save(os.path.join(input_directory, f.filename))
    img_array = preprocess_image(f.filename)
    pred = model.predict(img_array)
    label = np.argmax(pred[0])
    return render_template("result.html", pred = label)

if __name__ == '__main__':
    cnn.run(host = 'localhost', port = 5000)
    
