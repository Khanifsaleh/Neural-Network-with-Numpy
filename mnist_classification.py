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

def to_pixel(image, size):
    img_matrix = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img_matrix = cv2.resize(img_matrix, size)
    img_matrix = img_matrix.reshape(-1,1)/255
    return img_matrix

def converts(path_image, size):
    images = []
    labels = []
    list_image = os.listdir(path_image)

    for number in tqdm(list_image):
        imgs = os.listdir(os.path.join(path_image, number))
        for img in imgs:
            img_matrix = to_pixel(os.path.join(path_image, number, img), size)
            images.append(img_matrix)
            labels.append(one_hot_encode(int(number)))

    return images, labels

def train():
    path_image = os.path.join(path_this, 'dataset', 'mnist_png', 'training')
    images, labels = converts(path_image, size=(20,20))

    nn = NeuralNetwork()
    nn.add_input(20*20)
    nn.add_hidden(50, 'relu')
    nn.add_hidden(50, 'sigmoid')
    nn.add_output(10)

    nn.fit(images, labels, epochs=50, lr=0.001, batch_size=32, validation_split=0.25, shuffle=True)
    nn.save('./models/mnist_classification.pkl')

def test():
    path_image = os.path.join(path_this, 'dataset', 'mnist_png', 'testing')
    images, labels = converts(path_image, size=(20,20))
    
    nn = NeuralNetwork()
    nn.load('./models/mnist_classification.pkl')
    labels = [np.argmax(l) for l in labels]
    preds = np.array([nn.forward(img) for img in images])
    preds = np.array([np.argmax(p) for p in preds])
    print("test accuracy {} %".format(nn.accuracy(preds, labels) * 100))

if __name__ == '__main__':
    # train()
    test()