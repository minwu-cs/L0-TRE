"""
Neural network and image set related functions for MNIST

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""


import keras
from keras.models import model_from_json
from keras import backend as K
from keras.datasets import mnist
from keras.datasets import cifar10
import tensorflow as tf


def load_MNIST_neural_network():
    json_file = open('MNIST/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("MNIST/model.h5")
    print("MNIST neural network loaded from disk...")

    # compile loaded model
#   loaded_model.compile(loss='categorical_crossentropy',
#                        optimizer='adam',
#                        metrics=['accuracy'])

#   loaded_model.compile(loss='categorical_crossentropy',
#                        optimizer=keras.optimizers.Adadelta(),
#                        metrics=['accuracy'])

#    def fn(correct, predicted):
#        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
#                                                       logits=predicted)
#
#    loaded_model.compile(loss=fn,
#                         optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
#                         metrics=['accuracy'])

    loaded_model.summary()
    return loaded_model


'''
    x_test, y_test = obtain_test_images()
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)
    (loss, accuracy) = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
'''


def load_CIFAR10_neural_network():
    json_file = open('CIFAR10/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("CIFAR10/model.h5")
    print("MNIST neural network loaded from disk...")

    # compile loaded model
#   loaded_model.compile(loss='categorical_crossentropy',
#                        optimizer='adam',
#                        metrics=['accuracy'])

#   loaded_model.compile(loss='categorical_crossentropy',
#                        optimizer=keras.optimizers.Adadelta(),
#                        metrics=['accuracy'])

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    loaded_model.compile(loss=fn,
                         optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                         metrics=['accuracy'])

    loaded_model.summary()
    return loaded_model


def obtain_MNIST_test_images():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test / 255
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    return x_test, y_test


def obtain_CIFAR10_test_images():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test / 255

    return x_test, y_test


def retrieve_softmax_inputs(image, neural_network):
    # get inputs of softmax function, as softmax outputs, i.e., probability values,
    # may be too close to each other after just one pixel manipulation

#   func = K.function([neural_network.layers[0].input] + [K.learning_phase()],
#                     [neural_network.layers[neural_network.layers.__len__() - 1].output.op.inputs[0]])

    func = K.function([neural_network.layers[0].input] + [K.learning_phase()],
                      [neural_network.layers[neural_network.layers.__len__() - 1].output])

    softmax_inputs = func([image, 0])[0]
    return softmax_inputs