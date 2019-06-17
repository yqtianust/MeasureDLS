from __future__ import absolute_import

import torch
import torchvision

import numpy as np
import warnings

from ..models import *

VALID_DATASET_TYPE = ['MNIST']
VALID_DATASET_TYPE_STR = '/'.join(VALID_DATASET_TYPE)

class AccurancyMeasurer():

    def __init__(self, dataset_type):
        self.input_size = None
        self.loader = None 
        self.dataset = None
        self.dataset_type = None

        assert type(dataset_type) == str

        if dataset_type in VALID_DATASET_TYPE:
            self.dataset_type = dataset_type
        else:
            warnings.warn(
                'Please specify dataset_type as one of ' + VALID_DATASET_TYPE_STR)

        self._set_dataset()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_dataset(self):
        if self.dataset_type == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(
                root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
            self.loader = torch.utils.data.DataLoader(
                dataset=self.dataset, batch_size=10000, shuffle=False)
            self.input_size = 784
        

    def measure_accurancy(self, model):
        assert not (type(self.dataset_type) is None)
        assert self.dataset_type in VALID_DATASET_TYPE

        if type(model) == PyTorchModel:
            return self._measure_PyTorchModel_accurancy(model)
        else:
            pass 
        

    def _measure_PyTorchModel_accurancy(self, model):
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, labels) in enumerate(self.loader):
                images = images.reshape(-1, self.input_size).to(self.device).numpy()
                labels = labels.to(self.device).numpy()

                predictions = np.argmax(model.forward(images), axis=1)

                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)

                return accurancy
