"""
Image manipulation functions for MNIST

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""


from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from neural_network import *
from keras.preprocessing import image as Image
from numpy import linalg as LA


def L0Attack(image, neural_network, pixel_num, L0_Upper):
    img_label = neural_network.predict_classes(np.expand_dims(image, axis=0))
#   img_label = np.argmax(neural_network.predict(np.expand_dims(image, axis=0)))

    sorted_feature_map = influential_pixel_manipulation(image, neural_network, img_label, pixel_num)
    adv_image, adv_label, idx_first, success_flag = accumulated_pixel_manipulation(image, neural_network, sorted_feature_map, L0_Upper, img_label)

    if success_flag is True:
        adv_image, adv_label = refine_adversary_image(image, neural_network, adv_image, sorted_feature_map, idx_first, img_label)

    return adv_image, adv_label


# manipulate row*col pixels (:, :) at each time
def influential_pixel_manipulation(image, neural_network, max_index, pixel_num):
    # print('\nGetting influential pixel manipulations: ')

    new_pixel_list = np.linspace(0, 1, pixel_num)
    # print('Pixel manipulation: ', new_pixel_list)

    image_batch = np.kron(np.ones((pixel_num, 1, 1, 1)), image)

    manipulated_images = []
    (row, col, chl) = image.shape
    for i in range(0, row):
        for j in range(0, col):
            # need to be very careful about image.copy()
            changed_image_batch = image_batch.copy()
            for p in range(0, pixel_num):
                changed_image_batch[p, i, j, :] = new_pixel_list[p]
#           changed_image_batch[:, i, j] = np.array(new_pixel_list)
#           responseAllMat = np.concatenate((responseAllMat,changed_image_3d), axis=0)
            manipulated_images.append(changed_image_batch)  # each loop append [pixel_num, row, col, chl]
    manipulated_images = np.asarray(manipulated_images)     # [row*col, pixel_num, row, col, chl]
    # print('Manipulated images: ', manipulated_images.shape)

    manipulated_images = manipulated_images.reshape(row * col * pixel_num, row, col, chl)
    # print('Reshape dimensions to put into neural network: ', manipulated_images.shape)

    features_list = retrieve_softmax_inputs(manipulated_images, neural_network)
    # print('Softmax inputs: ', features_list.shape)

    feature_change = features_list[:, max_index].reshape(-1, pixel_num).transpose()
    # print('Softmax inputs of MaxIndex(', max_index, '): ', feature_change.shape)

    min_indices = np.argmin(feature_change, axis=0)
    min_values = np.amin(feature_change, axis=0)
    min_idx_values = min_indices.astype('float32') / (pixel_num - 1)

    [x, y] = np.meshgrid(np.arange(row), np.arange(col))
    x = x.flatten('F')  # to flatten in column-major order
    y = y.flatten('F')  # to flatten in column-major order

    target_feature_list = np.hstack((np.split(x, len(x)),
                                     np.split(y, len(y)),
                                     np.split(min_values, len(min_values)),
                                     np.split(min_idx_values, len(min_idx_values))))

    # a numpy array cannot have two types, e.g., int and float, simultaneously
#   target_feature_list[:, 0] = target_feature_list[:, 0].astype(int)
#   target_feature_list[:, 1] = target_feature_list[:, 1].astype(int)

    sorted_feature_map = target_feature_list[target_feature_list[:, 2].argsort()]
    # print('Sorted feature map: ', sorted_feature_map.shape)
    # print(sorted_feature_map)

    return sorted_feature_map


def accumulated_pixel_manipulation(image, neural_network, sorted_feature_map, mani_range, label):
    # print('\nLooking for adversary images...')
    # print('Manipulation range: ', mani_range)

    manipulated_images = []
    mani_image = image.copy()
    (row, col, chl) = image.shape
    for i in range(0, mani_range):
        # change row and col from 'float' to 'int'
        pixel_row = sorted_feature_map[i, 0].astype('int')
        pixel_col = sorted_feature_map[i, 1].astype('int')
        pixel_value = sorted_feature_map[i, 3]
        mani_image[pixel_row][pixel_col] = pixel_value
        # need to be very careful about image.copy()
        manipulated_images.append(mani_image.copy())

    manipulated_images = np.asarray(manipulated_images)     # [mani_range, row, col, chl]
    manipulated_labels = neural_network.predict_classes(manipulated_images)
#   manipulated_labels = neural_network.predict_classes(manipulated_images.reshape(len(manipulated_images), row, col, 1), verbose=0)
    # print('Manipulated images: ', manipulated_images.shape)
#    plt.imshow(manipulated_images[mani_range-1], cmap='Greys_r')
#    plt.show()
    # print('Manipulated labels: ', manipulated_labels.shape, '\n', manipulated_labels)

    adversary_images = manipulated_images[manipulated_labels != label, :, :, :]
    adversary_labels = manipulated_labels[manipulated_labels != label]
    # print('Adversary images: ', adversary_images.shape)
    # print('Adversary labels: ', adversary_labels.shape, '\n', adversary_labels)

    if adversary_labels.any():
        success_flag = True
        adversary_image = adversary_images[0]
        adversary_label = adversary_labels[0]
        idx_first = (manipulated_labels != label).nonzero()[0][0]
#       idx_first = np.amin((manipulated_labels != label).nonzero(), axis=1)
        # print('First adversary image found after', idx_first+1, 'pixel manipulations.')
    else:
        success_flag = False
        adversary_image = image
        adversary_label = label
        idx_first = np.nan
        # print('Adversary image not found.')

    return adversary_image, adversary_label, idx_first, success_flag


def refine_adversary_image(image, neural_network, adv_image_first, sorted_features, idx_first, label):
    # print('\nRefining found adversary image...')

    (row, col, chl) = image.shape
    refined_adversary = adv_image_first.copy()
    # print('Evaluating individual pixels: \nNo. ', end='')
    total_idx = 0
    idx_range = np.arange(idx_first)
    go_deeper = True
    while go_deeper is True:
        length = len(idx_range)
        for i in idx_range:
            pixel_row = sorted_features[i, 0].astype('int')
            pixel_col = sorted_features[i, 1].astype('int')
            refined_adversary[pixel_row, pixel_col] = image[pixel_row, pixel_col]
            refined_label = neural_network.predict_classes(np.expand_dims(refined_adversary, axis=0))
#           refined_label = neural_network.predict_classes(refined_adversary.reshape(1, row, col, 1), verbose=0)
            if refined_label == label:
                refined_adversary[pixel_row, pixel_col] = sorted_features[i, 3]
            else:
                total_idx = total_idx + 1
                idx_range = idx_range[~(idx_range == i)]
                # print(i+1, end=' ')
        if len(idx_range) == length:
            go_deeper = False
    # print('pixel(s) can be reverted.')
    # print('In total,', total_idx, 'pixel(s) reverted, leaving', idx_first+1-total_idx, 'pixel(s) manipulated.')

    refined_label = neural_network.predict_classes(np.expand_dims(refined_adversary, axis=0))
#   refined_label = neural_network.predict_classes(refined_adversary.reshape(1, row, col, 1), verbose=0)

#   if (refined_adversary == adv_image_first).all():
#       success_flag = 0
#   else:
#       success_flag = 1

    return refined_adversary, refined_label


def print_adversary_images(path, idx, refined_adversary, refined_class, image_diff, L0_distance):
#   title = path + 'idx_' + str(idx) + '_label_[' + str(predicted_class[0][1]) + '].png'
#   save_plot_image(image, title)

    title = path + 'idx_' + str(idx) + '_modified_label_[' + str(refined_class) + '].png'
    save_plot_image(refined_adversary, title)

#   title = path + 'idx_' + str(idx) + '_modified_diff_L0=' + str(L0_distance) + '_L1=' + str(L1_distance) + '_L2=' + str(L2_distance) + '.png'
    title = path + 'idx_' + str(idx) + '_modified_diff_L0=' + str(L0_distance) + '.png'
    save_plot_image(image_diff, title)

    print('\nOriginal image, refined adversarial image, and image difference saved in', path, 'directory.')


'''
def print_adversary_images(idx, image, label, refined_adversary, refined_label, image_diff, L0_distance, L1_distance, L2_distance):
    path = 'results/CIFAR10/'

    image = Image.array_to_img(image)
    refined_adversary = Image.array_to_img(refined_adversary)
    image_diff = Image.array_to_img(image_diff)

    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, interpolation='nearest')
#    ax.imshow(image * 255, cmap='Greys_r', interpolation='nearest')
#    ax.xaxis.set_ticks_position('bottom')
#    ax.yaxis.set_ticks_position('left')
    title = path + 'idx_' + str(idx) + '_label_' + str(label) + '.png'
    plt.savefig(title)

    ax.imshow(refined_adversary, interpolation='nearest')
#   ax.imshow(refined_adversary * 255, cmap='Greys_r', interpolation='nearest')
    title = path + 'idx_' + str(idx) + '_modified_label_' + str(refined_label) + '.png'
    plt.savefig(title)

    ax.imshow(image_diff, interpolation='nearest')
#   ax.imshow(image_diff * 255, cmap='Greys_r', interpolation='nearest')
    title = path + 'idx_' + str(idx) + '_modified_diff_L0=' + str(L0_distance) + '_L1=' + str(L1_distance) + '_L2=' + str(L2_distance) + '.png'
    plt.savefig(title)

    print('\nOriginal image, refined adversarial image, and image difference saved in /results/CIFAR10 directory.')
'''


''' 
    path = 'results/'
    title = path + 'idx_' + str(idx) + '_label_' + str(label) + '.png'
    plt.imsave(title, image * 255, cmap='Greys_r')
    title = path + 'idx_' + str(idx) + '_modified_label_' + str(refined_label) + '.png'
    plt.imsave(title, refined_adversary * 255, cmap='Greys_r')
    title = path + 'idx_' + str(idx) + '_modified_diff_L0=' + str(L0_distance) + '_L1=' + str(L1_distance) + '_L2=' + str(L2_distance) + '.png'
    plt.imsave(title, image_diff * 255, cmap='Greys_r')
'''


def save_plot_image(image, title):
    image = Image.array_to_img(image)

    plt.imsave(title, image)


'''    
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, interpolation='nearest')
#   ax.xaxis.set_ticks_position('bottom')
#   ax.yaxis.set_ticks_position('left')
    plt.savefig(title)

'''
