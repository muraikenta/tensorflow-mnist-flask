# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
import pdb
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

print('tensorflow: v', tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

target_images = 255 - np.array([
    np.array( Image.open('test_images/sneaker1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('test_images/t-shirt1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('test_images/sneaker2.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('test_images/trouser1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('test_images/pullover1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('test_images/bag1.jpg').resize((28, 28)).convert('L') ),
])
target_images = target_images / 255.0

target_lables = np.array([9, 0, 7, 1, 2, 8])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


def predict_test_images(images, labels):
    test_loss, test_acc = model.evaluate(images, labels)
    print(test_loss, test_acc)
    predictions = model.predict(images)

    # Plot the first 25 test images, their predicted label, and the true label
    # Color correct predictions in green, incorrect predictions in red
    plt.figure(figsize=(20,20))
    for i in range(min([len(images), 25])):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(images[i], cmap=plt.cm.binary)
        print(predictions[i])
        predicted_label = np.argmax(predictions[i])
        true_label = labels[i]
        if predicted_label == true_label:
          color = 'green'
        else:
          color = 'red'

        accuracy = round(float(predictions[i][predicted_label]), 4)

        label = "{} ({}) ({})".format(class_names[predicted_label], class_names[true_label], accuracy)
        plt.xlabel(label, color=color)

    plt.savefig('results/{}.png'.format(datetime.now().strftime('%s')))
    plt.show()

predict_test_images(target_images, target_lables)
# predict_test_images(test_images, test_labels)
