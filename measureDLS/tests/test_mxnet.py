import unittest

import mxnet
import numpy as np

import measureDLS.utils as utils
from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.mxnet import MXNetModel


class TestMXNet(unittest.TestCase):
    class ImageNetValDataset(mxnet.gluon.data.Dataset):

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        def __init__(self, preprocess):
            self._preprocess = preprocess
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
            self._y = mxnet.nd.array(self._y, dtype=int)

        def __len__(self):
            return len(self._y)

        def __getitem__(self, index):
            path = self._dir + '/' + self._filenames[index]
            x = mxnet.image.imread(path)
            x = mxnet.image.resize_short(x, 256)
            x, _ = mxnet.image.center_crop(x, (224, 224))
            x = x.transpose((2, 0, 1))
            x = x.astype('float32') / 255
            if self._preprocess:
                mean = mxnet.nd.array(self.mean).reshape(3, 1, 1)
                std = mxnet.nd.array(self.std).reshape(3, 1, 1)
                x = (x - mean) / std
            y = self._y[index].asscalar()
            return x, y

    def cifar10_dataset(self):
        dataset = mxnet.gluon.data.vision.CIFAR10(train=False)
        dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
        return dataset

    def mnist_dataset(self):
        dataset = mxnet.gluon.data.vision.MNIST(train=False)
        dataset._data = mxnet.nd.transpose(dataset._data.astype(np.float32), (0, 3, 1, 2)) / 255
        return dataset

    def test_imagenet_vgg16(self):
        model = mxnet.gluon.model_zoo.vision.vgg16(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.700000)
        self.assertAlmostEqual(accuracy.get(5), 0.950000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.597570, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_vgg19(self):
        model = mxnet.gluon.model_zoo.vision.vgg19(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.650000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.549809, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_resnet50_v2(self):
        model = mxnet.gluon.model_zoo.vision.resnet50_v2(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.775000)
        self.assertAlmostEqual(accuracy.get(5), 1.000000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.188258, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_mobilenet_v2(self):
        model = mxnet.gluon.model_zoo.vision.mobilenet_v2_1_0(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.725000)
        self.assertAlmostEqual(accuracy.get(5), 0.950000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.295334, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_cifar10_simple(self):
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
        bounds = (0, 1)
        num_classes = 10

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        self.assertAlmostEqual(accuracy.get(1), 0.687000)
        self.assertAlmostEqual(accuracy.get(5), 0.966600)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.470085, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_mnist_simple(self):
        model = mxnet.gluon.nn.Sequential()
        with model.name_scope():
            model.add(mxnet.gluon.nn.Dense(128, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(64, activation='relu'))
            model.add(mxnet.gluon.nn.Dense(10))
        model.load_parameters(utils.python_file_dir(__file__) + '/models/mxnet_mnist_simple.params')
        dataset = self.mnist_dataset()
        bounds = (0, 1)
        num_classes = 10

        measure_model = MXNetModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        self.assertAlmostEqual(accuracy.get(1), 0.978700)
        self.assertAlmostEqual(accuracy.get(5), 0.999700)
        self.assertAlmostEqual(neuron_coverage.get(0.8), 0.861386, places=6)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)


if __name__ == '__main__':
    unittest.main()
