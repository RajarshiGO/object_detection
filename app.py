
# from flask import Flask, render_template, request, flash, redirect, url_for
# import os
# import glob
# from werkzeug.utils import secure_filename
# import torch
# import cv2
# from PIL import Image
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
from flask import Flask, render_template, request
app = Flask(__name__)
app.debug = True
UPLOAD_FOLDER = './upload'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# acceptable_list = ['png', 'jpg']
# '''
@app.route('/')
def index():
    return render_template("index.html")
# '''
# @app.route('/', methods = ['GET'])
# def upload_return_file():
#     filelist = os.listdir(UPLOAD_FOLDER)
#     if (len(filelist) != 0):
#         for file in filelist:
#             os.remove(os.path.join(UPLOAD_FOLDER, file))
#     return render_template("index.html")
# @app.route('/uploader', methods = ['GET', 'POST'])
# def uploader():
#     response = request.get_json()
#     print(response)
#     json_file = open("file.json")
#     json_file.write(response)
#     f = request.files['file']
#     f.save(secure_filename(f.filename))
#     filelist = os.listdir(UPLOAD_FOLDER)
#     if(len(filelist) != 0):
#         for file in filelist:
#             ext = file.split('.')[1]
#             if(ext in acceptable_list):
#                 img = Image.open(os.path.join(UPLOAD_FOLDER, file))
#                 result = model(img)
#                 result.save()
#     return render_template("page.html")

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/api/upload', methods =['POST'])
def upload():
    f = request.files['file']
    filepath = f'static/temp/{f.filename}'
    print(f.filename)
    f.save(filepath)
    return f"{request.url_root}{filepath}"
if __name__ == '__main__':
    app.run(debug=True)
