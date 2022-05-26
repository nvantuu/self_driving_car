import os

import pandas as pd
from auto_drive import config as c


def read_df(df_path=None):
    return pd.read_csv(c.log_path)

def generate_samples(df:pd.DataFrame=None):
    list_samples = []

    centerImgsName = df['center'].values
    leftImgsName = df['left'].values
    rightImgsName = df['right'].values
    angleSteerings = df['steering'].values
    print(centerImgsName[0])

    for i in range(len(df)):
        center_angleSteering = angleSteerings[i]
        left_angleSteering = angleSteerings[i] + 0.2
        right_angleSteering = angleSteerings[i] - 0.2

        list_samples.append((centerImgsName[i], center_angleSteering))
        list_samples.append((leftImgsName[i], left_angleSteering))
        list_samples.append((rightImgsName[i], right_angleSteering))

    return list_samples

import cv2
import torch
from torch.utils.data import Dataset

class SelfDrivingCarData(Dataset):

    def __init__(self, samples):
        self.df = samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(c.data_path, str(self.df[idx][0]).strip())
        # print(img_path)
        img = cv2.imread( img_path )
        # img = load_image(img_path)
        img = cv2.resize(img[-150:], (70, 320)) / 255.0
        # print(img.shape) (320, 70, 3)
        img = torch.from_numpy(img).float()
        # print(img.size()) torch.Size([320, 70, 3])

        # corresponding steering angle
        label = self.df[idx][1]

        return img, label



from torch.utils.data import random_split

def data_split(samples):
    train_size = round(len(samples) * c.TRAINING_RATIO)
    val_size = len(samples) - train_size
    print(len(samples), train_size, val_size)
    train_samples, val_samples = random_split(samples, [train_size, val_size])

    return train_samples, val_samples



