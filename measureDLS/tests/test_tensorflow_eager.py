from __future__ import print_function

import math
import unittest
import warnings

import tensorflow as tf
import numpy as np

import measureDLS.utils as utils
from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.tensorflow_eager import TensorFlowEagerModel


class TestTensorFlowEager(unittest.TestCase):
    class ImageNetValDataGenerator(tf.keras.utils.Sequence):

        def __init__(self, batch_size=32, transform=None):
            self._batch_size = batch_size
            self._transform = transform
            self._dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
            self._filenames = []
            self._y = []
            with open(self._dir + '/' + 'ILSVRC2012_validation_ground_truth.txt', 'r') as f:
                lines = f.readlines()
            for line in lines:
                splits = line.split('---')
                if len(splits) != 5:
                    continue
                self._filenames.append(splits[0])
                self._y.append(int(splits[2]))
            self._y = np.array(self._y, dtype=int)

        def __len__(self):
            return math.ceil(len(self._y) / self._batch_size)

        def __getitem__(self, index):
            start = index * self._batch_size
            end = (index + 1) * self._batch_size
            filenames = self._filenames[start:end]
            num = len(filenames)
            x = np.empty((num, 224, 224, 3), dtype=np.float32)
            for i in range(num):
                filename = filenames[i]
                path = self._dir + '/' + filename
                image = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path, target_size=(224, 224)))
                if self._transform is not None:
                    image = self._transform(image)
                x[i] = image
            y = self._y[start:end]
            return x, y

    def cifar10_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        y_train = y_train.flatten().astype('int32')
        x_test = x_test.astype('float32') / 255
        y_test = y_test.flatten().astype('int32')
        return x_test, y_test

    def mnist_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = y_train.flatten().astype('int32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_test = y_test.flatten().astype('int32')
        return x_test, y_test

    def test_imagenet_vgg16(self):
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.VGG16()
        model.trainable = False
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_generator_preprocess = self.ImageNetValDataGenerator(transform=lambda x:(x[..., ::-1] - mean) / std)
        data_generator_original = self.ImageNetValDataGenerator(batch_size=1)
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict_generator(data_generator_preprocess, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs_generator(data_generator_preprocess, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples_generator(data_generator_original, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], preprocessing=(mean, std))

        self.assertAlmostEqual(accuracy.get(1), 0.575000)
        self.assertAlmostEqual(accuracy.get(5), 0.775000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.554487, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_vgg19(self):
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.VGG19()
        model.trainable = False
        mean = (103.939, 116.779, 123.68)
        std = (1, 1, 1)
        data_generator_preprocess = self.ImageNetValDataGenerator(transform=lambda x:(x[..., ::-1] - mean) / std)
        data_generator_original = self.ImageNetValDataGenerator(batch_size=1)
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict_generator(data_generator_preprocess, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs_generator(data_generator_preprocess, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples_generator(data_generator_original, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], preprocessing=(mean, std))

        self.assertAlmostEqual(accuracy.get(1), 0.550000)
        self.assertAlmostEqual(accuracy.get(5), 0.825000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.505920, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_resnet50_v2(self):
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.ResNet50V2()
        model.trainable = False
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_generator_preprocess = self.ImageNetValDataGenerator(transform=lambda x: (x - mean) / std)
        data_generator_original = self.ImageNetValDataGenerator(batch_size=1)
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict_generator(data_generator_preprocess, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs_generator(data_generator_preprocess, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples_generator(data_generator_original, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], preprocessing=(mean, std))

        self.assertAlmostEqual(accuracy.get(1), 0.625000)
        self.assertAlmostEqual(accuracy.get(5), 0.875000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.452325, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_mobilenet_v2(self):
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.applications.MobileNetV2()
        model.trainable = False
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_generator_preprocess = self.ImageNetValDataGenerator(transform=lambda x: (x - mean) / std)
        data_generator_original = self.ImageNetValDataGenerator(batch_size=1)
        bounds = (0, 255)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict_generator(data_generator_preprocess, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs_generator(data_generator_preprocess, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples_generator(data_generator_original, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], preprocessing=(mean, std))

        self.assertAlmostEqual(accuracy.get(1), 0.675000)
        self.assertAlmostEqual(accuracy.get(5), 0.825000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.296583, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_cifar10_simple(self):
        tf.keras.backend.set_learning_phase(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            model = tf.keras.models.load_model(utils.python_file_dir(__file__) + '/models/keras_cifar10_simple.h5')
        model.trainable = False
        x, y = self.cifar10_data()
        bounds = (0, 1)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(x[:5], y[:5], 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples])

        self.assertAlmostEqual(accuracy.get(1), 0.777400)
        self.assertAlmostEqual(accuracy.get(5), 0.985000)
        self.assertAlmostEqual(neuron_coverage.get(0.8), 0.710084, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_mnist_simple(self):
        tf.keras.backend.set_learning_phase(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            model = tf.keras.models.load_model(utils.python_file_dir(__file__) + '/models/keras_mnist_simple.h5')
        model.trainable = False
        x, y = self.mnist_data()
        bounds = (0, 1)

        measure_model = TensorFlowEagerModel(model)

        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(x[:5], y[:5], 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples])

        self.assertAlmostEqual(accuracy.get(1), 0.991000)
        self.assertAlmostEqual(accuracy.get(5), 1.000000)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.589744, places=2)
        self.assertAlmostEqual(robustness.success_rate, 0.6666666667)


if __name__ == '__main__':
    unittest.main()
