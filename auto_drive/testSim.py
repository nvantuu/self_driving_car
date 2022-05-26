import os

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask
import numpy as np


import cv2
print("cv2")

import eventlet
print("...")

import socketio

import base64

from io import BytesIO
print("io")

from PIL import Image

from auto_drive.model import AutoPilotModel
import auto_drive.config as c

print("...")

sio = socketio.Server()
app = Flask(__name__)  # '__main__'
model = AutoPilotModel()
print("...")

# maxSpeed = 15

MAX_SPEED = 15
MIN_SPEED = 10

speed_limit = MAX_SPEED


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (70, 320)) / 255.0
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    print(f'speed: {speed}')
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    global speed_limit
    if speed > speed_limit:
        speed_limit = MIN_SPEED  # slow down
    else:
        speed_limit = MAX_SPEED
    throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
    print(f'{steering_angle}, {throttle}, {speed}')
    sendControl(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    path_save = os.path.join(c.save_path, 'model_1_epoch.pth')
    model.load_state_dict(torch.load(path_save))

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    print("app initialized successfully")

    ### LISTEN TO PORT 4567
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    print("terminal")


