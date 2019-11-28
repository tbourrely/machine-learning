from os import path
import sys

from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils

# Process mnist dataset and return only test data
def mnistTestData():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_dimension = 784
    num_classes = 10

    x_test = x_test.reshape(10000, input_dimension)
    x_test = x_test.astype('float32')
    x_test /= 255

    y_test = np_utils.to_categorical(y_test, num_classes)

    return (x_test, y_test)

# Evaluate a model and print the score data
def evaluateModel(model, x_test, y_test, verbose=0):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

def quitIfFileNotExist(file):
    if not path.exists(bestModelPath):
        print(bestModelPath + ' does not exists')
        sys.exit(1)

#====================================================================

bestModelPath = 'models/best-model.hdf5'

quitIfFileNotExist(bestModelPath)
model = load_model(bestModelPath)
(x_test, y_test) = mnistTestData()
evaluateModel(model, x_test, y_test)

