from PIL import Image
import numpy as np

my_test_images = 255 - np.array([
    np.array( Image.open('my_test_images/sneaker1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('my_test_images/t-shirt1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('my_test_images/sneaker2.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('my_test_images/trouser1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('my_test_images/pullover1.jpg').resize((28, 28)).convert('L') ),
    np.array( Image.open('my_test_images/bag1.jpg').resize((28, 28)).convert('L') ),
])

my_test_lables = np.array([9, 0, 7, 1, 2, 8])
