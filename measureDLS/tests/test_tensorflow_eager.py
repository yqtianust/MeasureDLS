from __future__ import print_function

import os
import time
import unittest

import numpy as np
import tensorflow as tf

import measureDLS.utils as utils
from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.tensorflow_eager import TensorFlowEagerModel


class TestTensorFlowEager(unittest.TestCase):

    def imagenet_dataset(self):
        dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
        files = sorted(list(filter(lambda file: file.lower().endswith('.jpeg'), os.listdir(dir))))
        num = len(files)
        x = np.empty((num, 224, 224, 3), dtype=np.float32)
        y = np.empty((num,), dtype=int)
        for i in range(num):
            path = dir + '/' + files[i]
            image = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path, target_size=(224, 224)))
            x[i] = image
            y[i] = int(files[i].split('_')[-1].split('.')[0])
        return x, y

    def cifar10_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        y_train = y_train.flatten().astype('int32')
        x_test = x_test.astype('float32') / 255
        y_test = y_test.flatten().astype('int32')
        return x_test, y_test

    def mnist_dataset(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = y_train.flatten().astype('int32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_test = y_test.flatten().astype('int32')
        return x_test, y_test

    def test_imagenet_vgg16(self):
        print('*' * 20)
        print('test_imagenet_vgg16')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.VGG16()
        model.trainable = False
        dataset_original = self.imagenet_dataset()
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        x_preprocessed = (x_original[..., ::-1] - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.7)
        self.assertAlmostEqual(accuracy_top_5.get, 0.9)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.3)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.25424865831842575)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_vgg19(self):
        print('*' * 20)
        print('test_imagenet_vgg19')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.VGG19()
        model.trainable = False
        dataset_original = self.imagenet_dataset()
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        x_preprocessed = (x_original[..., ::-1] - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.65)
        self.assertAlmostEqual(accuracy_top_5.get, 0.85)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.3)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.19080021774632552)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_resnet50_v2(self):
        print('*' * 20)
        print('test_imagenet_resnet50_v2')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.ResNet50V2()
        model.trainable = False
        dataset_original = self.imagenet_dataset()
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.6)
        self.assertAlmostEqual(accuracy_top_5.get, 0.75)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.37823457716107256)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_mobilenet_v2(self):
        print('*' * 20)
        print('test_imagenet_mobilenet_v2')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.MobileNetV2()
        model.trainable = False
        dataset_original = self.imagenet_dataset()
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.75)
        self.assertAlmostEqual(accuracy_top_5.get, 0.95)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.5)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.12128607809847199)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_cifar10_simple(self):
        print('*' * 20)
        print('test_cifar10_simple')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(utils.python_file_dir(__file__) + '/models/keras_cifar10_simple.h5')
        model.trainable = False
        dataset = self.cifar10_dataset()
        x = dataset[0]
        y_true = dataset[1]
        dataset_small = (x[:5], y_true[:5])
        bounds = (0, 1)
        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.7774)
        self.assertAlmostEqual(accuracy_top_5.get, 0.985)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.8)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.12044817927170869)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_mnist_simple(self):
        print('*' * 20)
        print('test_mnist_simple')

        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(utils.python_file_dir(__file__) + '/models/keras_mnist_simple.h5')
        model.trainable = False
        dataset = self.mnist_dataset()
        x = dataset[0]
        y_true = dataset[1]
        dataset_small = (x[:5], y_true[:5])
        bounds = (0, 1)
        measure_model = TensorFlowEagerModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.991)
        self.assertAlmostEqual(accuracy_top_5.get, 1)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.6)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.5213675213675214)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 0.4)
        end = time.time()
        print('robustness time', end - start)


if __name__ == '__main__':
    unittest.main()
