# Demo Prediction 
import tensorflow as tf
import numpy as np
import cv2

IMAGE_SIZE = 240
# Define the path to the saved model
model_path = 'D:/model'

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Define the image path
image_path = r'C:/Users/psaik/Desktop/Amanita_Muscaria_P.jpg'

# Read the image
image = cv2.imread(image_path)

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
print('The predicted class is', predicted_class_name)
print(f"The predicted class is {class_index}")