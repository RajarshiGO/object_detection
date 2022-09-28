from flask import Flask, render_template, request
import tensorflow as tf
from utils import custom_loss
model = tf.keras.models.load_model('model', custom_objects = {"custom_loss": custom_loss})
app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = './upload'

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/api/upload', methods =['POST'])
def upload():
    f = request.files['file']
    filepath = f'static/temp/{f.filename}'
    print(f.filename)
    f.save(filepath)
    return f"{request.url_root}{filepath}"
if __name__ == '__main__':
    app.run(debug=True)
