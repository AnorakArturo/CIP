import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import os


IMAGE_SIZE = 240
# Define the path to the saved model
drive_url = os.environ.get('drive_url')
model_path = 'drive_url'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Resize the image to the size expected by the model
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Preprocess the image by scaling its pixel values between 0 and 1
    image_normalized = image_resized / 255.0

    # Add an extra dimension to the image to represent the batch size (1 in this case)
    image_batch = np.expand_dims(image_normalized, axis=0)

    # Use the model to make a prediction on the image batch
    prediction = model.predict(image_batch)

    # Get the class with the highest probability
    class_index = np.argmax(prediction)

    # Print the predicted class
    class_names = ['edible', 'poisonous']
    predicted_class_name = class_names[class_index]

    # Return the predicted class as a JSON response
    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
