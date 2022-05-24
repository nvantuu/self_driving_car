import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import batch_generator

if __name__ == '__main__':
    data_df = pd.read_csv('dataset/driving_log.csv',
                          names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    data_df['center'] = data_df['center'].apply(lambda x: 'dataset/IMG/' + x.split('/')[-1])
    data_df['left'] = data_df['left'].apply(lambda x: 'dataset/IMG/' + x.split('/')[-1])
    data_df['right'] = data_df['right'].apply(lambda x: 'dataset/IMG/' + x.split('/')[-1])

    data_df['steering'] = data_df['steering'].apply(lambda x: str(x).strip())

    X = data_df[['center', 'left', 'right']].values
    X = X[1:]

    y = data_df['steering'].values
    y = y[1:]
    y = y.astype(np.float)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


    def createModel():
        net = keras.Sequential()

        net.add(keras.layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
        net.add(keras.layers.Conv2D(24, (5, 5), (2, 2), activation='elu'))
        net.add(keras.layers.Conv2D(36, (5, 5), (2, 2), activation='elu'))
        net.add(keras.layers.Conv2D(48, (5, 5), (2, 2), activation='elu'))
        net.add(keras.layers.Conv2D(64, (3, 3), activation='elu'))
        net.add(keras.layers.Conv2D(64, (3, 3), activation='elu'))

        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(100, activation='elu'))
        net.add(keras.layers.Dense(50, activation='elu'))
        net.add(keras.layers.Dense(10, activation='elu'))
        net.add(keras.layers.Dense(1, activation='tanh'))

        net.compile(tf.optimizers.Adam(lr=1.0e-4), loss='mse')
        return net

    best_model = keras.callbacks.ModelCheckpoint('best_model.h5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

    model = createModel()
    model.fit_generator(batch_generator(X_train, y_train, 40, True),
                        steps_per_epoch=2000,
                        validation_data=batch_generator(X_valid, y_valid, 40, False),
                        validation_steps=10,
                        epochs=1,
                        callbacks=[best_model, tensorboard_callback]
                        )
