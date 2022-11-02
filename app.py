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
model = tf.keras.models.load_model('model', custom_objects = {"custom_loss": custom_loss})
app = Flask(__name__)
app.config['SECRET_KEY'] = "Rajarshi"
UPLOAD_FOLDER = './static/upload'
preprocess = albu.Compose([
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
])
@app.route('/')
def index():
    return render_template("index.html")

class LinkForm(FlaskForm):
    link = StringField("Please provide a link below:", validators = [DataRequired(), URL()])
    submit = SubmitField("Submit")

@app.route('/api/upload', methods =['POST'])
def upload():
    files = os.listdir(UPLOAD_FOLDER)
    if(len(files)!=0):
        for file in files:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    f = request.files['file']
    filepath = f'{UPLOAD_FOLDER}/{f.filename}'
    f.save(filepath)
    image = Image.open(filepath)
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
    drawn = draw_bboxes(filepath, bboxes = pred)
    imsave(filepath, drawn)
    return f"{request.url_root}{filepath}"

@app.route('/prediction', methods =['GET', 'POST'])
def get_pred_link():
    image_path = None
    link = None
    form = LinkForm()
    if form.validate_on_submit():
        link = form.link.data
        form.link.data = ""
        files = os.listdir(UPLOAD_FOLDER)
        if(len(files)!=0):
            for file in files:
                os.remove(os.path.join(UPLOAD_FOLDER, file))
        r = requests.get(link, allow_redirects=True)
        open(f'{UPLOAD_FOLDER}/image.jpg', 'wb').write(r.content)
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
        imsave(f'{UPLOAD_FOLDER}/image.jpg', drawn)
        image_path = f'{UPLOAD_FOLDER}/image.jpg'
    return render_template("link.html", image_path = image_path, form = form)

@app.route('/instructions', methods = ['GET'])
def show_instructions():
    return render_template("instructions.html")

if __name__ == '__main__':
    app.run(debug=True)
