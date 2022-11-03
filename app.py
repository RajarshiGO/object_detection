# Import the necessary libraries and also the  helper functions from utils.py
from flask import Flask, render_template, request
from matplotlib.pyplot import imsave
import tensorflow as tf
from utils import custom_loss
import os
from PIL import Image
import albumentations as albu
import numpy as np
from utils import process_predictions, draw_bboxes, form_image_grid
import requests
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, URL

# Import our saved object-detection model
model = tf.keras.models.load_model('model', custom_objects = {"custom_loss": custom_loss})

# Define our app instance and define a secrect key
app = Flask(__name__)
app.config['SECRET_KEY'] = "Rajarshi"

# Set the folder to store user input images
UPLOAD_FOLDER = './static/upload'

# Convert the input image to gray scale and enhance contrast
preprocess = albu.Compose([
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
])

# Add the default route to our homepage
@app.route('/')
def index():
    return render_template("index.html")

# Define a Flask-WTForm instance for the URL input form
class LinkForm(FlaskForm):
    link = StringField("Please provide a link below:", validators = [DataRequired(), URL()])
    submit = SubmitField("Submit")

# Define the object-detection API
@app.route('/api/upload', methods =['POST'])
def upload():
    # First empty the contents of UPLOAD_FOLDER
    files = os.listdir(UPLOAD_FOLDER)
    if(len(files)!=0):
        for file in files:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    #Get the uploaded file and save it ti UPLOAD_FOLDER
    f = request.files['file']
    filepath = f'{UPLOAD_FOLDER}/{f.filename}'
    f.save(filepath)
    # Open the uploaded image
    image = Image.open(filepath)
    # Get the image dimensions for scaling the predicted boxes later
    width, height = image.size
    # Resize the image
    image = image.resize((256,256))
    # Convert the image data to numpy array and apply the preprocessing
    image = np.array(image)
    image = preprocess(image = image)
    image = image['image']/255
    image = np.expand_dims(image, axis = 0)
    # Get prediction from the model
    pred = model.predict(image)
    pred = np.squeeze(pred)
    # Process predictions from the model and convert them to the required format
    pred = process_predictions(pred, form_image_grid())
    # Scale the predicted boxes to the original image dimensions
    pred[:, 0] = pred[:, 0]/256 * width
    pred[:, 1] = pred[:, 1]/256 * height
    pred[:, 2] = pred[:, 2]/256 * width
    pred[:, 3] = pred[:, 3]/256 * height
    # Draw bounding boxes over the input image before returning it back
    drawn = draw_bboxes(filepath, bboxes = pred)
    imsave(filepath, drawn)
    return f"{request.url_root}{filepath}"

@app.route('/prediction', methods =['GET', 'POST'])
def get_pred_link():
    # Initialize form credentials
    image_path = None
    link = None
    form = LinkForm()
    # Validate form submission and then empty the contents of UPLOAD_FOLDER
    if form.validate_on_submit():
        link = form.link.data
        form.link.data = ""
        files = os.listdir(UPLOAD_FOLDER)
        if(len(files)!=0):
            for file in files:
                os.remove(os.path.join(UPLOAD_FOLDER, file))
        # Download image in the specified URL and save it to UPLOAD_FOLDER
        r = requests.get(link, allow_redirects=True)
        open(f'{UPLOAD_FOLDER}/image.jpg', 'wb').write(r.content)
        # Rest of the code is same as previous section
        image = Image.open(f'{UPLOAD_FOLDER}/image.jpg')
        width, height = image.size
        image = image.resize((256,256))
        image = np.array(image)
        image = preprocess(image = image)
        image = image['image']/255
        image = np.expand_dims(image, axis = 0)
        pred = model.predict(image)
        pred = np.squeeze(pred)
        pred = process_predictions(pred, form_image_grid())
        pred[:, 0] = pred[:, 0]/256 * width
        pred[:, 1] = pred[:, 1]/256 * height
        pred[:, 2] = pred[:, 2]/256 * width
        pred[:, 3] = pred[:, 3]/256 * height
        drawn = draw_bboxes(f'{UPLOAD_FOLDER}/image.jpg', bboxes = pred)
        # This time instead of returning the image, we save it instead replacing the input input image
        imsave(f'{UPLOAD_FOLDER}/image.jpg', drawn)
        image_path = f'{UPLOAD_FOLDER}/image.jpg'
    # Return the html page that hosts the URL form, output image path and the form object
    return render_template("link.html", image_path = image_path, form = form)

@app.route('/instructions', methods = ['GET'])
def show_instructions():
    return render_template("instructions.html")

if __name__ == '__main__':
    app.run(debug=True)
