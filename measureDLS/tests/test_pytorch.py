import unittest

import PIL
import numpy as np
import torch
import torchvision

import measureDLS.utils as utils
from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.pytorch import PyTorchModel


class TestPyTorch(unittest.TestCase):
    class ImageNetValDataset(torch.utils.data.Dataset):

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
            self._y = torch.LongTensor(self._y)

        def __len__(self):
            return len(self._y)

        def __getitem__(self, index):
            path = self._dir + '/' + self._filenames[index]
            x = PIL.Image.open(path)
            x = x.convert('RGB')
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
            x = transforms(x)
            if self._preprocess:
                x = torchvision.transforms.Normalize(mean=self.mean, std=self.std)(x)
            y = self._y[index]
            return x, y

    def cifar10_dataset(self):
        dataset = torchvision.datasets.CIFAR10(root=utils.python_file_dir(__file__) + '/data', train=False, download=True)
        x = torch.from_numpy(dataset.data / 255.0).permute(0, 3, 1, 2).float()
        y = torch.tensor(dataset.targets)
        dataset = torch.utils.data.dataset.TensorDataset(x, y)
        return dataset

    def mnist_dataset(self):
        dataset = torchvision.datasets.MNIST(root=utils.python_file_dir(__file__) + '/data', train=False, download=True)
        x = torch.unsqueeze((dataset.data / 255.0), 1)
        y = dataset.targets
        dataset = torch.utils.data.dataset.TensorDataset(x, y)
        return dataset

    def test_imagenet_vgg16(self):
        model = torchvision.models.vgg16(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.650000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.597570, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_vgg19(self):
        model = torchvision.models.vgg19(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.700000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.550014, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_resnet50_v2(self):
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.775000)
        self.assertAlmostEqual(accuracy.get(5), 1.000000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.103692, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_mobilenet_v2(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        dataset_preprocessed = self.ImageNetValDataset(True)
        dataset_original = self.ImageNetValDataset(False)
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset_preprocessed, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset_preprocessed, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(np.array(self.ImageNetValDataset.mean).reshape((3, 1, 1)), np.array(self.ImageNetValDataset.std).reshape((3, 1, 1))))

        self.assertAlmostEqual(accuracy.get(1), 0.675000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.273926, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_cifar10_simple(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3)
                self.pool1 = torch.nn.MaxPool2d(2)
                self.conv2 = torch.nn.Conv2d(32, 64, 3)
                self.pool2 = torch.nn.MaxPool2d(2)
                self.conv3 = torch.nn.Conv2d(64, 64, 3)
                self.fc1 = torch.nn.Linear(64 * 4 * 4, 64)
                self.fc2 = torch.nn.Linear(64, 10)

            def forward(self, data):
                data = torch.nn.functional.relu(self.conv1(data))
                data = self.pool1(data)
                data = torch.nn.functional.relu(self.conv2(data))
                data = self.pool2(data)
                data = torch.nn.functional.relu(self.conv3(data))
                data = data.view(-1, 64 * 4 * 4)
                data = torch.nn.functional.relu(self.fc1(data))
                data = self.fc2(data)
                return data

        model = Model()
        model.load_state_dict(torch.load(utils.python_file_dir(__file__) + '/models/pytorch_cifar10_simple.pth'))
        dataset = self.cifar10_dataset()
        bounds = (0, 1)
        num_classes = 10

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        self.assertAlmostEqual(accuracy.get(1), 0.427500)
        self.assertAlmostEqual(accuracy.get(5), 0.906400)
        self.assertAlmostEqual(neuron_coverage.get(0.6), 0.534188, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_mnist_simple(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(1 * 28 * 28, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 10)

            def forward(self, data):
                data = data.view(-1, 1 * 28 * 28)
                data = torch.nn.functional.relu(self.fc1(data))
                data = torch.nn.functional.relu(self.fc2(data))
                data = self.fc3(data)
                return data

        model = Model()
        model.load_state_dict(torch.load(utils.python_file_dir(__file__) + '/models/pytorch_mnist_simple.pth'))
        dataset = self.mnist_dataset()
        bounds = (0, 1)
        num_classes = 10

        measure_model = PyTorchModel(model)

        accuracy = Accuracy()
        measure_model.predict(dataset, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(dataset, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset, 3, bounds, num_classes, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        self.assertAlmostEqual(accuracy.get(1), 0.962200)
        self.assertAlmostEqual(accuracy.get(5), 0.998900)
        self.assertAlmostEqual(neuron_coverage.get(0.7), 0.876238, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)


if __name__ == '__main__':
    unittest.main()
