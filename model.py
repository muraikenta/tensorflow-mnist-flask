# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Helper libraries
# from datetime import datetime
# import numpy as np
# import matplotlib.pyplot as plt

# my modules
# from my_test_images import my_test_images, my_test_lables

def build_model():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), _ = fashion_mnist.load_data()

    train_images = preprocess_data(train_images)
    # test_images = test_images / 255.0
    # my_test_images = my_test_images / 255.0

    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)

    return model

# TODO: チェッカーの実験
# def preprocess_data(images: np.ndarray) -> np.ndarray:
def preprocess_data(images: np.ndarray) -> np.ndarray:
    return images / 255.0
