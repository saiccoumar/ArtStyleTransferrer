
from flask import Response
from flask import Flask
from flask import render_template
import tensorflow as tf
import os
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time
from PIL import Image
import numpy as np
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

current_directory = os.getcwd()
print(current_directory)
statDir = current_directory+'/static/'
templateDir = current_directory+'/templates/'

# initialize a flask object
app = Flask(__name__,static_folder=statDir,
            template_folder=templateDir)

# add list and iteration through style path
content_path = current_directory+"/selfie.JPG"
style_path = current_directory+"/kandinskiy3.jpg"

# 256 to 512 dimensions appear to be the most effective at creating art
def scale(image, max_dim = 256):
  # Casts a tensor to a new type.
  original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

  # Creates a scale constant for the image
  scale_ratio = max_dim / max(original_shape)

  # Casts a tensor to a new type.
  new_shape = tf.cast(original_shape * scale_ratio, tf.int32)

  # Resizes the image based on the scaling constant generated above
  return tf.image.resize(image, new_shape)

def load_img(path_to_img):

  # Reads and outputs the entire contents of the input filename.
  img = tf.io.read_file(path_to_img)

  # Detect whether an image is a BMP, GIF, JPEG, or PNG, and 
  # performs the appropriate operation to convert the input 
  # bytes string into a Tensor of type dtype
  img = tf.image.decode_image(img, channels=3)

  # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
  img = tf.image.convert_image_dtype(img, tf.float32)

  # Scale the image using the custom function we created
  img = scale(img)

  # Adds a fourth dimension to the Tensor because
  # the model requires a 4-dimensional Tensor
  return img[tf.newaxis, :]

def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

content_image = load_img(content_path)
style_image = load_img(style_path)

# %matplotlib inline
def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)

# plt.figure(figsize=(12, 12))
# plt.subplot(1, 2, 1)
# plt.imshow(content_image[0])
# plt.title('Content Image')
# plt.subplot(1, 2, 2)
# plt.imshow(style_image[0])
# plt.title('Style Image')

# plt.show()



# Load Magenta's Arbitrary Image Stylization network from TensorFlow Hub  
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

print("checkpoint 1")
# Pass content and style images as arguments in TensorFlow Constant object format
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
print("checkpoint 2")


IMAGE_PATH = stylized_image
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

hr_image = preprocess_image(IMAGE_PATH)
plot_image(tf.squeeze(hr_image), title="Original Image")
model = hub.load(SAVED_MODEL_PATH)
start = time.time()
fake_image = model(hr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))
plot_image(tf.squeeze(fake_image), title="Super Resolution")
# save_image(tf.squeeze(hr_image), filename="Original Image")
# Set the size of the plot figure
# plt.figure(figsize=(12, 12))

# # Plot the stylized image
# plt.imshow(stylized_image[0])

# # Add title to the plot
# plt.title('Stylized Image')

# # Hide axes
# plt.axis('off')

# # Show the plot
# plt.show()

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	app.run(port=8000, debug=True, use_reloader=False)
