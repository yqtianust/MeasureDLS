from __future__ import absolute_import

import torch
import torchvision

import numpy as np
import warnings

from ..models import *

VALID_DATASET_TYPE = ['MNIST', 'CIFAR10', 'IMAGENET']
VALID_DATASET_TYPE_STR = '/'.join(VALID_DATASET_TYPE)

class AccurancyMeasurer():

    def __init__(self, dataset_type, is_input_flatten=True):
        self.is_input_flatten = is_input_flatten

        self.loader = None
        self.dataset = None
        self.dataset_type = None

        assert type(dataset_type) == str

        if dataset_type in VALID_DATASET_TYPE:
            self.dataset_type = dataset_type
        else:
            warning_str = 'Please specify dataset_type as one of the following datasets: ' + VALID_DATASET_TYPE_STR
            warnings.warn(warning_str)

        self._set_dataset()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _set_dataset(self):
        if self.dataset_type == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(
                root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
            self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset, batch_size=10000, shuffle=False)
        elif self.dataset_type == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
            self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset, batch_size=10000, shuffle=False)
        elif self.dataset_type == 'IMAGENET':
            self.dataset = torchvision.datasets.ImageNet(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
            self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset, batch_size=10000, shuffle=False)
        else:
            pass 

    def measure_accurancy(self, model):
        assert not (type(self.dataset_type) is None)
        assert self.dataset_type in VALID_DATASET_TYPE

        if type(model) == PyTorchModel:
            if self.dataset_type == 'MNIST':
                return self._measure_PyTorchModel_MNIST_accurancy(model)
        else:
            pass

    def _measure_PyTorchModel_MNIST_accurancy(self, model):
        with torch.no_grad():
            correct = 0
            total = 0

            for _, (images, labels) in enumerate(self.loader):
                if self.is_input_flatten:
                    images = images.reshape(-1, 784).to(self.device).numpy()
                else:
                    images = images.reshape(-1, 1, 28, 28).to(self.device).numpy()

                labels = labels.to(self.device).numpy()
                predictions = np.argmax(model.forward(images), axis=1)

                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)

                return accurancy
