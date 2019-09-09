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

from ..dataset_prepare import load_imagenet_val_dataset 

class AccurancyMeasurer():

    def __init__(self, dataset_type, transform=None, is_input_flatten=True, preprocess=None):
        """
        Three tasks are accomplished at the initialization of an AccurancyMeasurer instance:
        1. Check whether dataset_type given from instatitation is valid (has been implemented)
        2. Initialize several interal parameters 
        3. Check if GPU avaiable
        """
        # Internal Hyperparamters
        self.batch_size = 1000

        # User-defined Hyperparamters
        self.transform = transform
        self.is_input_flatten = is_input_flatten
        self.preprocess = preprocess

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
        """
        Check whether the required dataset is appropriately downloaded. 

        If NOT, we will automatically download required dataset (for dataset of small-scale) or give proper instructions (for dataset of huge-scale like ImageNet). 
        """
        if self.dataset_type == 'MNIST':
            self.dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        elif self.dataset_type == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=self.transform, download=True)
            self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        elif self.dataset_type == 'IMAGENET':
            # self.dataset = torchvision.datasets.ImageNet(root='./data', split='val',transform=self.transform, download=True)
            # self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1000, shuffle=False)
            
            # Check the existence of 'x_val.npy' and 'y_val.npy' 
            # If required files are not found, raise FileNotFoundError 
            import os, errno # lazy import 
            file_path = 'data/x_val.npy'
            if not os.path.exists(file_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
            file_path = 'data/y_val.npy'
            if not os.path.exists(file_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

            # Otherwise, load some data
            num_of_samples = 1000
            x, y = load_imagenet_val_dataset(num_of_samples)
            x = np.transpose(x, (0, 3, 1, 2))

            import torchvision.transforms as transforms # lazy import 
            IMAGENET_TRANSFORM = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

            tensor_x = torch.from_numpy(x)
            tensor_x = IMAGENET_TRANSFORM(tensor_x)
            
            tensor_y = torch.from_numpy(y)
            self.dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=num_of_samples) # create your dataloader

    def _set_Keras_dataset(self):
        """
        Check whether the required dataset is appropriately downloaded. 

        If NOT, we adopt the similar strategy as mentioned in _set_Pytorch_dataset function. 
        """
        if self.dataset_type == 'MNIST':
            _, self.dataset = mnist.load_data()
        elif self.dataset_type == 'CIFAR10':
            pass 
        elif self.dataset_type == 'IMAGENET':
            self.dataset = load_imagenet_val_dataset(1000)

    def _set_TensorFlow_dataset(self):
        """
        Check whether the required dataset is appropriately downloaded.

        If NOT, we adopt the similar strategy as mentioned in _set_Pytorch_dataset function.
        """
        if self.dataset_type == 'MNIST':
            _, self.dataset = mnist.load_data()
            print(self.dataset)
        elif self.dataset_type == 'CIFAR10':
            pass
        elif self.dataset_type == 'IMAGENET':
            self.dataset = load_imagenet_val_dataset(1000)

    def measure_accurancy(self, model):
        assert not (type(self.dataset_type) is None)
        assert self.dataset_type in VALID_DATASET_TYPE

        if type(model) == PyTorchModel:
            self._set_Pytorch_dataset()
            return self._measure_PyTorchModel_accurancy(self.dataset_type, model)
        elif type(model) == KerasModel:
            self._set_Keras_dataset()
            return self._measure_KerasModel_accurancy(self.dataset_type, model)
        elif type(model) == TensorFlowModel:
            self._set_TensorFlow_dataset()
            return self._measure_TensorFlowModel_accurancy(self.dataset_type, model)
        else:
            pass

    def _measure_PyTorchModel_accurancy(self, dataset_type, model):
        with torch.no_grad():
            correct = 0
            total = 0

            for _, (inputs, labels) in enumerate(self.loader):
                if self.is_input_flatten:
                    if dataset_type == 'MNIST':
                        inputs = inputs.reshape(-1, 784).to(self.device).cpu().numpy()
                    elif dataset_type == 'CIFAR10':
                        inputs = inputs.reshape(-1, 3*32*32).to(self.device).cpu().numpy()
                    elif dataset_type == 'IMAGENET':
                        inputs = inputs.reshape(-1, 3*224*224).to(self.device).cpu().numpy()
                else:
                    if dataset_type == 'MNIST':
                        inputs = inputs.reshape(-1, 1, 28, 28).to(self.device).cpu().numpy()
                    elif dataset_type == 'CIFAR10':
                        inputs = inputs.to(self.device).cpu().numpy()
                    elif dataset_type == 'IMAGENET':
                        inputs = inputs.to(self.device).cpu().numpy()

                print(inputs.shape, labels.shape)
                labels = labels.to(self.device).cpu().numpy()
                predictions = np.argmax(model.forward(inputs), axis=1)

                total += labels.shape[0]
                correct += (predictions == labels).sum().item()
                accurancy = (correct/total)
                
                return accurancy


    def _measure_KerasModel_accurancy(self, dataset_type, model):
        datas, labels = self.dataset
        if not (self.preprocess is None):
            datas = self.preprocess(datas)
        self.dataset = (datas, labels)
        acc = model.evaluate(self.dataset)
        return acc 

    def _measure_TensorFlowModel_accurancy(self, dataset_type, model):
        datas, labels = self.dataset
        if not (self.preprocess is None):
            datas = self.preprocess(datas)
        self.dataset = (datas, labels)
        acc = model.evaluate(self.dataset)
        return acc

