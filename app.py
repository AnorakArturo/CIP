from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Path to the saved model
model_path = 'D:/model'
model = tf.keras.models.load_model(model_path)


# Define a function to preprocess the image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [240, 240])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, (1, 240, 240, 3))  # reshape to (1, 240, 240, 3) to match model input shape
    return image


# Define a function to make a prediction
def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make a prediction
    prediction = model.predict(preprocessed_image)
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    return predicted_class


# Define a route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')


# Define a route for predicting on uploaded images
@app.route('/predict', methods=['POST'])
def predict_image():
    # Get the image from the request
    image = request.files['image'].read()
    # Make a prediction
    predicted_class = predict(image)
    # Return the predicted class as JSON
    return {'predicted_class': predicted_class}


if __name__ == '__main__':
    app.run(debug=True)
