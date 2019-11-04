import os

import matplotlib.pyplot as plt
import numpy as np


def python_file_dir(file):
    return os.path.dirname(os.path.abspath(file))


def to_numpy(data):
    if 'mxnet' in str(type(data)):
        data = data.asnumpy()
    elif 'torch' in str(type(data)):
        data = data.cpu().numpy()
    elif 'numpy' in str(type(data)):
        data = np.copy(data)
    return data


def draw_adversarial_samples(origins, adversarials):
    for i in range(len(origins)):
        origin = origins[i]
        adversarial = adversarials[i]
        if (origin == adversarial).all():
            continue
        origin = _draw_image_preprocessing(origin)
        adversarial = _draw_image_preprocessing(adversarial)
        plt.subplot(1, 3, 1)
        plt.imshow(origin)
        plt.subplot(1, 3, 2)
        plt.imshow(adversarial)
        plt.subplot(1, 3, 3)
        plt.imshow(adversarial - origin)
        plt.show()


def _draw_image_preprocessing(image):
    shape = image.shape
    shape_len = len(shape)
    if shape_len == 1 and shape[0] == 784:
        image = image.reshape(28, 28)
        shape_len = 2
    if shape_len == 2:
        image = np.expand_dims(image, axis=0)
        shape_len = 3
    transpose = [v for v in range(shape_len)]
    min_index = 0
    for i in range(shape_len):
        if image.shape[i] < image.shape[min_index]:
            min_index = i
    transpose.remove(min_index)
    transpose.append(min_index)
    transpose = tuple(transpose)
    image = np.transpose(image, transpose)
    shape = image.shape
    if shape[-1] == 1:
        shape = list(shape)
        shape[-1] = 3
        image_new = np.empty(shape)
        image_new[..., 0] = image_new[..., 1] = image_new[..., 2] = image[..., 0]
        image = image_new
    if image.max() > 1:
        image /= 255.0
    return image
