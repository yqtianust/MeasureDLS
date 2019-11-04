import os
import time
import unittest

import mxnet
import numpy as np

import measureDLS.utils as utils
from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.mxnet import MXNetModel


class TestMXNet(unittest.TestCase):

    def imagenet_dataset_mean(self):
        return (0.485, 0.456, 0.406)

    def imagenet_dataset_std(self):
        return (0.229, 0.224, 0.225)

    def imagenet_dataset(self, preprocess):
        dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
        files = sorted(list(filter(lambda file: file.lower().endswith('.jpeg'), os.listdir(dir))))
        num = len(files)
        x = mxnet.nd.empty((num, 3, 224, 224))
        y_true = mxnet.nd.empty((num,), dtype=int)
        for i in range(num):
            path = dir + '/' + files[i]
            image = mxnet.image.imread(path)
            image = mxnet.image.resize_short(image, 256)
            image, _ = mxnet.image.center_crop(image, (224, 224))
            image = image.transpose((2, 0, 1)).expand_dims(axis=0)
            image = image.astype('float32') / 255
            if preprocess:
                mean = mxnet.nd.array(self.imagenet_dataset_mean()).reshape(1, 3, 1, 1)
                std = mxnet.nd.array(self.imagenet_dataset_std()).reshape(1, 3, 1, 1)
                image = (image - mean) / std
            x[i] = image
            y_true[i] = int(files[i].split('_')[-1].split('.')[0])
        dataset = mxnet.gluon.data.ArrayDataset(x, y_true)
        return dataset

    def cifar10_dataset(self):
        dataset = mxnet.gluon.data.vision.CIFAR10(train=False)
        dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
        return dataset

    def mnist_dataset(self):
        dataset = mxnet.gluon.data.vision.MNIST(train=False)
        dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
        return dataset

    def test_imagenet_vgg16(self):
        print('*' * 20)
        print('test_imagenet_vgg16')

        model = mxnet.gluon.model_zoo.vision.vgg16(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = mxnet.gluon.data.ArrayDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.85)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.95)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.18209600477042337)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_resnet50_v2(self):
        print('*' * 20)
        print('test_imagenet_resnet50_v2')

        model = mxnet.gluon.model_zoo.vision.resnet50_v2(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = mxnet.gluon.data.ArrayDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.75)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 1)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.2)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.1782314402577189)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_mobilenet_v2(self):
        print('*' * 20)
        print('test_imagenet_mobilenet_v2')

        model = mxnet.gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = mxnet.gluon.data.ArrayDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.75)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.85)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.5)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.10725342768686422)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_cifar10_simple(self):
        print('*' * 20)
        print('test_cifar10_simple')

        model = mxnet.gluon.nn.Sequential()
        with model.name_scope():
            model.add(mxnet.gluon.nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.MaxPool2D(pool_size=2))
            model.add(mxnet.gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.MaxPool2D(pool_size=2))
            model.add(mxnet.gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
            model.add(mxnet.gluon.nn.Flatten())
            model.add(mxnet.gluon.nn.Dense(64, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(10))
        model.load_parameters(utils.python_file_dir(__file__) + '/models/mxnet_cifar10_simple.params')
        dataset = self.cifar10_dataset()
        x = dataset[:][0]
        y_true = dataset[:][1]
        dataset_small = mxnet.gluon.data.ArrayDataset(x[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 10

        measure_model = MXNetModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.687)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.9666)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.6)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.2948717948717949)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_mnist_simple(self):
        print('*' * 20)
        print('test_mnist_simple')

        model = mxnet.gluon.nn.Sequential()
        with model.name_scope():
            model.add(mxnet.gluon.nn.Dense(128, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(64, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(10))
        model.load_parameters(utils.python_file_dir(__file__) + '/models/mxnet_mnist_simple.params')

        dataset = self.mnist_dataset()
        x = dataset[:][0]
        y_true = dataset[:][1]
        dataset_small = mxnet.gluon.data.ArrayDataset(x[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 10

        measure_model = MXNetModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.9787)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.9997)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.8)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.3465346534653465)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)


if __name__ == '__main__':
    unittest.main()
