from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# My modules
from model import build_model, preprocess_data
from class_names import class_names

def run(model, images, labels):
    test_loss, test_acc = model.evaluate(images, labels)
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

    plt.savefig('test_results/{}.png'.format(datetime.now().strftime('%s')))
    plt.show()

_, (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

test_images = preprocess_data(test_images)
model = build_model()

run(model, test_images, test_labels)
