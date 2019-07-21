from __future__ import absolute_import

import torch
import torchvision
import torchvision.transforms as transforms

from keras.datasets import mnist
from keras.utils import np_utils 

import numpy as np
import warnings

from ..models import *
from ..constants import VALID_DATASET_TYPE, VALID_DATASET_TYPE_STR

class AccurancyMeasurer():

    def __init__(self, dataset_type, transform=None, is_input_flatten=True):
        # Internal Hyperparamters
        self.batch_size = 1000

        # User-defined Hyperparamters
        self.transform = transform
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_Pytorch_dataset(self):
        if self.dataset_type == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        elif self.dataset_type == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=self.transform, download=True)
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        elif self.dataset_type == 'IMAGENET':
            self.dataset = torchvision.datasets.ImageNet(root='./data', split='val',transform=self.transform, download=True)
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1000, shuffle=False)

    def _set_Keras_dataset(self):
        if self.dataset_type == 'MNIST':
            _, self.dataset = mnist.load_data()
            print(self.dataset)
        elif self.dataset_type == 'CIFAR10':
            pass 
        elif self.dataset_type == 'IMAGENET':
            pass 

    def measure_accurancy(self, model):
        assert not (type(self.dataset_type) is None)
        assert self.dataset_type in VALID_DATASET_TYPE

        if type(model) == PyTorchModel:
            self._set_Pytorch_dataset()
            if self.dataset_type == 'MNIST':
                return self._measure_PyTorchModel_MNIST_accurancy(model)
            elif self.dataset_type == 'CIFAR10':
                return self._measure_PyTorchModel_CIFAR10_accurancy(model)
            elif self.dataset_type == 'IMAGENET':
                return self._measure_PyTorchModel_IMAGENET_accurancy(model)
        elif type(model) == KerasModel:
            self._set_Keras_dataset()
            if self.dataset_type == 'MNIST':
                return self._measure_KerasModel_MNIST_accurancy(model)
            elif self.dataset_type == 'CIFAR10':
                return self._measure_KerasModel_CIFAR10_accurancy(model)
            elif self.dataset_type == 'IMAGENET':
                return self._measure_KerasModel_IMAGENET_accurancy(model)
        else:
            pass

    def _measure_PyTorchModel_MNIST_accurancy(self, model):
        with torch.no_grad():
            correct = 0
            total = 0

            for _, (inputs, labels) in enumerate(self.loader):
                if self.is_input_flatten:
                    inputs = inputs.reshape(-1, 784).to(self.device).numpy()
                else:
                    inputs = inputs.reshape(-1, 1, 28, 28).to(self.device).numpy()

                labels = labels.to(self.device).numpy()
                predictions = np.argmax(model.forward(inputs), axis=1)

                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)

                return accurancy

    def _measure_PyTorchModel_CIFAR10_accurancy(self, model):
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (inputs, labels) in enumerate(self.loader):
                if self.is_input_flatten:
                    inputs = inputs.reshape(-1, 3*32*32).to(self.device).numpy()
                else:
                    inputs = inputs.to(self.device).numpy()

                labels = labels.to(self.device).numpy()
                predictions = np.argmax(model.forward(inputs), axis=1)
                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)
                
                return accurancy

    def _measure_PyTorchModel_IMAGENET_accurancy(self, model):
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (inputs, labels) in enumerate(self.loader):
                if self.is_input_flatten:
                    inputs = inputs.reshpae(-1, 3*224*224).to(self.device).numpy()
                else:
                    inputs = inputs.to(self.device).numpy()

                labels = labels.to(self.device).numpy()
                predictions = np.argmax(model.forward(inputs), axis=1)
                total += labels.shape[0]

                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)

                return accurancy

    def _measure_KerasModel_MNIST_accurancy(self, model):
        pass 

    def _measure_KerasModel_CIFAR10_accurancy(self, model):
        pass 

    def _measure_KerasModel_IMAGENET_accurancy(self, model):
        pass 