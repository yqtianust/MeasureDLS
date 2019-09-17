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

    def __init__(self, dataset_type, transform=None, is_input_flatten=True, preprocess=None, k_degree=1):
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
        self.k_degree = k_degree

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
            ''' DEBUG code segment, which should be abandoned
 
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
            x, y = load_imagenet_val_dataset(self.num_of_samples)
            x = np.transpose(x, (0, 3, 1, 2))

            tensor_x = torch.from_numpy(x)
            tensor_y = torch.from_numpy(y)

            import torchvision.transforms.functional as F # lazy import
            normalized_tensor_x = torch.empty(tensor_x.shape)
            for i, sample in enumerate(tensor_x):
                sample = sample /255
                normalized_sample = F.normalize(sample, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                normalized_tensor_x[i] = normalized_sample
            
            self.dataset = torch.utils.data.TensorDataset(normalized_tensor_x, tensor_y) # create your datset
            '''
            
            # Use ImageFolder here 
            self.dataset = torchvision.datasets.ImageFolder('data/val', transform=self.transform) 
            self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size) # create your dataloader

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

    def _measure_top_k_accurancy(self, y_true, y_pred, k=1):
        argsorted_y = np.argsort(y_pred)[:,-k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

    def _measure_KerasModel_accurancy(self, dataset_type, model):
        datas, labels = self.dataset
        if not (self.preprocess is None):
            datas = self.preprocess(datas)
        
        from keras.utils import to_categorical #lazy import 
        labels_one_hot = to_categorical(labels, 1000)
        predictions = model.forward(datas)
        top_k_acc = self._measure_top_k_accurancy(labels_one_hot, predictions, k=self.k_degree)
        
        '''
        self.dataset = (datas, labels)
        acc = model.evaluate(self.dataset)
        '''
        return top_k_acc

    def _measure_TensorFlowModel_accurancy(self, dataset_type, model):
        datas, labels = self.dataset
        if not (self.preprocess is None):
            datas = self.preprocess(datas)
        self.dataset = (datas, labels)
        acc = model.evaluate(self.dataset)
        return acc

