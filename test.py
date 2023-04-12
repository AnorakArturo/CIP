import tensorflow as tf
print(tf.__version__)

# # Path to the saved model directory
# model_dir = 'D:/model'

# # Load the saved model
# tf.compat.v1.disable_eager_execution() # disable eager execution
# loaded_model = tf.keras.models.load_model(model_dir, compile=False)


# # Get the path to the saved_model.pbtxt file
# pbtxt_path = '{}/saved_model.pbtxt'.format(model_dir)

# # Read the contents of the saved_model.pbtxt file
# with open(pbtxt_path, 'r') as f:
#     pbtxt_contents = f.read()

# # Get the TensorFlow version from the saved_model.pbtxt file
# tf_version_start = pbtxt_contents.find('tensorflow_version: ') + len('tensorflow_version: ')
# tf_version_end = pbtxt_contents.find('\n', tf_version_start)
# tf_version = pbtxt_contents[tf_version_start:tf_version_end]

# # Print the TensorFlow version
# print('TensorFlow version used to save the model: {}'.format(tf_version))
