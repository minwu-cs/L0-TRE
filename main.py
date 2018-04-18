"""
Main file for MNIST/CIFAR10

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""


from __future__ import print_function
from matplotlib import pyplot as plt
from numpy import linalg as LA
import time
import pickle
from l0attack import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

idx_min = 0
idx_max = 2

pixel_num = 2
mani_range = 100
# mani_range = row * col

dataset = 'CIFAR10'

if dataset is 'MNIST':
    NN = load_MNIST_neural_network()
    (x_test, y_test) = obtain_MNIST_test_images()
    classes = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
    result_path = 'results/MNIST/'
elif dataset is 'CIFAR10':
    NN = load_CIFAR10_neural_network()
    (x_test, y_test) = obtain_CIFAR10_test_images()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    result_path = 'results/CIFAR10/'

(row, col, chl) = x_test[0].shape
print('\n' * 2)
print('L0 Attack begins: ')

# idx_first_list = []

Results = []
for idx in range(idx_min, idx_max):
    print('Image index: ', idx)
    image = x_test[idx]
    (row, col, chl) = image.shape
#   label = MNIST.predict_classes(image.reshape(1, row, col, 1), verbose=0)
    label = NN.predict_classes(np.expand_dims(image, axis=0))
    print('Image size: ', image.shape)
    print('Image label: ', label)

    title = result_path + 'idx_' + str(idx) + '_label_[' + str(classes[label[0]]) + '].png'
    save_plot_image(image, title)

    tic = time.time()
    adv_image, adv_label = L0Attack(image, NN, pixel_num, mani_range)
    elapsed = time.time() - tic
    print('\nElapsed time: ', elapsed)

    image_diff = np.abs(adv_image - image)
    # note pixels have been transformed from [0,255] to [0,1]
    L0_distance = int((image_diff * 255 > 1).sum() / chl)
    #   L1_distance = image_diff.sum()
    #   L2_distance = LA.norm(image_diff)

    Results.append([idx, image, label, adv_image, adv_label, image_diff, L0_distance, elapsed])

    print_adversary_images(result_path, idx,
                           adv_image, classes[adv_label[0]],
                           image_diff, L0_distance)
#       print_adversary_images(idx, image, classes[label[0]],
#                              refined_adversary, classes[refined_label[0]],
#                              image_diff, L0_distance, L1_distance, L2_distance)

    print('---------------------------------------------------------------------------', '\n' * 5)

with open(result_path + dataset + '.dat', 'wb') as f:
    pickle.dump(Results, f)

