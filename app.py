from flask import Flask, render_template, request
from matplotlib.pyplot import imsave
import tensorflow as tf
from utils import custom_loss
import os
from PIL import Image
import albumentations as albu
import numpy as np
from utils import process_predictions, draw_bboxes, form_image_grid
model = tf.keras.models.load_model('model', custom_objects = {"custom_loss": custom_loss})
app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = './static/upload'
preprocess = albu.Compose([
    albu.CLAHE(p=1),
    albu.ToGray(p=1),
])
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/upload', methods =['POST'])
def upload():
    files = os.listdir(UPLOAD_FOLDER)
    if(len(files)!=0):
        for file in files:
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    f = request.files['file']
    filepath = f'{UPLOAD_FOLDER}/{f.filename}'
    print(f.filename)
    f.save(filepath)
    image = Image.open(filepath)
    image = image.resize((256,256))
    image = np.array(image)
    image = preprocess(image = image)
    image = image['image']/255
    image = np.expand_dims(image, axis = 0)
    pred = model.predict(image)
    pred = np.squeeze(pred)
    pred = process_predictions(pred, form_image_grid())
    drawn = draw_bboxes(filepath, bboxes = pred)
    imsave(filepath, drawn)
    return f"{request.url_root}{filepath}"
if __name__ == '__main__':
    app.run(debug=True)
