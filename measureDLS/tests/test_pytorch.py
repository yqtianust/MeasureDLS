import os
import time
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

    def imagenet_dataset_mean(self):
        return (0.485, 0.456, 0.406)

    def imagenet_dataset_std(self):
        return (0.229, 0.224, 0.225)

    def imagenet_dataset(self, preprocess):
        dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
        files = sorted(list(filter(lambda file: file.lower().endswith('.jpeg'), os.listdir(dir))))
        num = len(files)
        x = torch.empty((num, 3, 224, 224), dtype=torch.float32)
        y = torch.empty((num,), dtype=torch.int64)
        for i in range(num):
            path = dir + '/' + files[i]
            input_image = PIL.Image.open(path)
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
            input_tensor = transforms(input_image)
            if preprocess:
                input_tensor = torchvision.transforms.Normalize(mean=self.imagenet_dataset_mean(), std=self.imagenet_dataset_std())(input_tensor)
            x[i] = input_tensor
            y[i] = int(files[i].split('_')[-1].split('.')[0])
        dataset = torch.utils.data.dataset.TensorDataset(x, y)
        return dataset

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
        print('*' * 20)
        print('test_imagenet_vgg16')

        model = torchvision.models.vgg16(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = torch.utils.data.dataset.TensorDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.8)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.9)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.3)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.27437388193202145)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_resnet50_v2(self):
        print('*' * 20)
        print('test_imagenet_resnet50_v2')

        model = torchvision.models.wide_resnet50_2(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = torch.utils.data.dataset.TensorDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.9)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 1)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.1)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.13627130794965747)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_imagenet_mobilenet_v2(self):
        print('*' * 20)
        print('test_imagenet_mobilenet_v2')

        model = torchvision.models.mobilenet_v2(pretrained=True)
        dataset_preprocessed = self.imagenet_dataset(True)
        x_preprocessed = dataset_preprocessed[:][0]
        y_true = dataset_preprocessed[:][1]
        dataset_original = self.imagenet_dataset(False)
        x_original = dataset_original[:][0]
        dataset_original_small = torch.utils.data.dataset.TensorDataset(x_original[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 1000

        measure_model = PyTorchModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.8)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.95)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.2)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.22153300841825432)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples], preprocessing=(np.array(self.imagenet_dataset_mean()).reshape(3, 1, 1), np.array(self.imagenet_dataset_std()).reshape(3, 1, 1)))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_cifar10_simple(self):
        print('*' * 20)
        print('test_cifar10_simple')

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
        x = dataset[:][0]
        y_true = dataset[:][1]
        dataset_small = torch.utils.data.dataset.TensorDataset(x[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 10

        measure_model = PyTorchModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.4275)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.9064)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.6)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.3888888888888889)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_mnist_simple(self):
        print('*' * 20)
        print('test_mnist_simple')

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
        x = dataset[:][0]
        y_true = dataset[:][1]
        dataset_small = torch.utils.data.dataset.TensorDataset(x[:5], y_true[:5])
        bounds = (0, 1)
        num_classes = 10

        measure_model = PyTorchModel(model)

        start = time.time()
        accuracy_top_1 = Accuracy(k=1)
        accuracy_top_5 = Accuracy(k=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.accuracy, 0.9622)
        self.assertAlmostEqual(accuracy_top_5.accuracy, 0.9989)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.7)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.neuron_coverage, 0.5495049504950495)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, num_classes, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)


if __name__ == '__main__':
    unittest.main()
