
from flask import Response
from flask import Flask, request, flash, redirect, url_for
from flask import render_template
import tensorflow as tf
import os
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import random
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')

current_directory = os.getcwd()
print(current_directory)
statDir = current_directory+'/static/'
templateDir = current_directory+'/templates/'
artistList=['kandinskiy','monet','nagai','banksy','dali','kahlo','gogh','picasso']





# initialize a flask object
app = Flask(__name__,static_folder=statDir,
            template_folder=templateDir)

# add list and iteration here
content_path = current_directory+"/IMG_7080.jpg"



# 256 to 512 dimensions appear to be the most effective at creating art
def scale(image, max_dim = 512):
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


# plt.figure(figsize=(12, 12))
# plt.subplot(1, 2, 1)
# plt.imshow(content_image[0])
# plt.title('Content Image')
# plt.subplot(1, 2, 2)
# plt.imshow(style_image[0])
# plt.title('Style Image')

# plt.show()
def convertAll(image):
  for i in artistList:
    convert(image,i)


# Load Magenta's Arbitrary Image Stylization network from TensorFlow Hub  
# premade models sometimes get updated and need to be replaced.
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

print("checkpoint 1")
def convert(image,artist):
  kandinskiy_style = current_directory+"/art"+"/"+artist+str(random.randint(1,3) )+".jpg"
  content_image = load_img(image)
  style_image = load_img(kandinskiy_style)
# Pass content and style images as arguments in TensorFlow Constant object format
  stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
  print("checkpoint 2:"+kandinskiy_style)

# Set the size of the plot figure
  plt.figure(figsize=(6, 6))

# Plot the stylized image
  plt.imshow(stylized_image[0])

  # Add title to the plot
  # plt.title('Stylized Image')

# Hide axes
  plt.axis('off')

# Show the plot
  plt.savefig(current_directory+"/static/output/"+artist+"Output.jpg")
# convertAll(content_path)
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

  

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
        # check if the post request has the file part
      if 'file' not in request.files:
          flash('No file part')
          return redirect(request.url)
      file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
      if file.filename == '':
          flash('No selected file')
          return redirect(request.url)
      if file and file.filename:
        filename = secure_filename(file.filename)
        file.save(current_directory+'/uploads/'+filename)
        for r in artistList:
          convert(("uploads/"+filename),r)
          print(r+" converted")
        for f in os.listdir('uploads'):
          os.remove(os.path.join('uploads', f))
        return render_template("images.html")
  return render_template("index.html")

# converts one file to banksy
# convert("selfie.jpg","banksy")
  

# check to see if this is the main thread of execution
if __name__ == '__main__':
  app.run(port=8000, debug=True, use_reloader=True)
