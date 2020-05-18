import os
import sys
from tqdm import tqdm
import cv2
import numpy as np

path_this = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path_this)

from neural_network import NeuralNetwork

def one_hot_encode(index, num_labels=10):
    zeros = np.zeros((num_labels, 1))
    zeros[index] = 1
    return np.array(zeros)

def to_pixel(path_image):
    images = []
    labels = []
    list_image = os.listdir(path_image)

    for number in tqdm(list_image):
        imgs = os.listdir(os.path.join(path_image, number))
        for img in imgs:
            img_matrix = cv2.imread(os.path.join(path_image, number, img), cv2.IMREAD_GRAYSCALE)
            img_matrix = cv2.resize(img_matrix, (20, 20))
            img_matrix = img_matrix.reshape(-1,1)/255
            images.append(img_matrix)
            labels.append(one_hot_encode(int(number)))

    return images, labels

path_image = os.path.join(path_this, 'dataset', 'mnist_png', 'training')
images, labels = to_pixel(path_image)

nn = NeuralNetwork()
nn.add_input(20*20)
nn.add_hidden(50, 'relu')
nn.add_hidden(50, 'sigmoid')
nn.add_output(10)

nn.fit(images, labels, epochs=40, lr=0.0001, validation_split=0.25, shuffle=True)