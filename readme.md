
# A YOLO based object detection model using a pre-trained ResNet50 backbone from scratch

![demo](demo.gif)
This project demonstrates how to develop a object detection deep learning model over a pre-trained CNN backbone such as ResNet50 from sratch. The model is trained on the [Global Wheat detection](https://www.kaggle.com/c/global-wheat-detection) challenge data available on [kaggle](kaggle.com). The entire deep learning model is written and trained in tensorflow and numpy and then deployed as web application using flask and [Apache httpd](https://httpd.apache.org/) server and then containerized using [docker[(https://www.docker.com/). The web app can be viewed [here](https://wheat-tip-detection.azurewebsites.net/).

Check out the ```notebook.ipynb``` jupyter notebook for step by step explanation.

# To run locally using docker
1. Install docker using your distribution's package manager or follow the instructions on the official [website.](https://docs.docker.com/engine/install/)
2. Open a terminal and pull the image from docker-hub using the following command.
   
    ```docker pull rajarshi13g/object-detection```
3. Run a container using the pulled image and also bind the port 8080 with the container.
   
    ```docker run -p8080:80 rajarshi13g/object-detection```
4. Open a browser window and browse to this address ```localhost:8080```.

## Run locally without docker

1. Clone this repo.
   
    ```git clone https://github.com/RajarshiGO/object_detection```
2. Open a termial and change the working directory to the cloned repository.
   
    ```cd object_detection```
3.  Install the dependecies through ```pip```. Make sure to install ```pip``` beforehand.
   
    ```pip install -r requirements.txt```
4. Type the command below to start a flask server.
   
   ```flask run```
5. Open a browser window and go to ```localhost:5000``` to launch the app.
