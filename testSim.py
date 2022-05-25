import keras

print("...")

import os
print("...")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("...")

# from tensorflow.keras.models import load_model
from keras.models import load_model
print("...")

from flask import Flask
print("...")

import numpy as np
print("...")

import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
print("cv2")

import eventlet
print("...")

import socketio
print("...")

import base64
print("...")

from io import BytesIO
print("io")

from PIL import Image

print("...")

sio = socketio.Server()
app = Flask(__name__)  # '__main__'
model = None
print("...")

maxSpeed = 15


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    print(f'speed: {speed}')
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print(f'{steering}, {throttle}, {speed}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    },skip_sid=True)


if __name__ == '__main__':
    print(os.getcwd())
    # model_path = os.path.join(os.getcwd(), 'best_model_1_epoch.h5')
    model_path = os.path.join(os.getcwd(), 'best_model_20_epoch.h5')
    # model_path = os.path.join(os.getcwd(), 'model.h5')
    print(model_path)
    print("OK")

    model = keras.models.load_model(model_path)
    print("model loaded successfully")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    print("app initialized successfully")

    ### LISTEN TO PORT 4567
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    print("terminal")


