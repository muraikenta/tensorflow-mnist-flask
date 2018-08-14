# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_model(epochs=1):
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), _ = fashion_mnist.load_data()

    train_images = preprocess_data(train_images)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=epochs)

    return model

# TODO: typeチェッカーの実験
def preprocess_data(images: np.ndarray) -> np.ndarray:
    return images / 255.0
