from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.utils import to_categorical

model2 = tf.keras.models.load_model('combined_model.h5')

app = Flask(__name__)

def process_image(image, img_size=(128, 128), color_mode='L'):
    img = Image.open(image).convert(color_mode) 
    img = img.resize(img_size) 
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = img_array.reshape(1, img_size[0], img_size[1], 1)  
    return img_array

def process_integer(input_data):
    input_array = np.array(input_data, dtype=np.int32).reshape(1, 64)  
    return input_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        integer_data = request.form.getlist('integer_data')  
        
        integer_data = list(map(int, integer_data))  
        
        img_array = process_image(image_file) 
        integer_array = process_integer(integer_data)  

        prediction = model2.predict([img_array, integer_array])
        predicted_class = np.argmax(prediction, axis=1)

        return jsonify({'predicted_class': int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True)
