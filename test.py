import warnings
import unittest

import measureDLS
import torch 
from torchvision import models
from torchvision import transforms

class TestMeasureDLS(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)

    def test_sample_model_Pytorch_FC_MNIST_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_FC_MNIST()    
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='MNIST', is_input_flatten=True)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.92)

    def test_sample_model_Pytorch_CNN_MNIST_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_CNN_MNIST()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='MNIST', is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.98)


    def test_sample_model_Pytorch_FC_CIFAR10_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_FC_CIFAR10()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='CIFAR10', is_input_flatten=True)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.47)

    def test_sample_model_Pytorch_CNN_CIFAR10_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_CNN_CIFAR10()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='CIFAR10', is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.6)

    def test_pretrained_model_Pytorch_Alexnet_IMAGENET_accurancy(self):
        alexnet = models.alexnet(pretrained=True)
        user_model = alexnet.eval()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='IMAGENET', is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.6)

    def test_pretrained_model_Pytorch_Resnet18_IMAGENET_accurancy(self):
        resnet = models.resnet18(pretrained=True)
        user_model = resnet.eval()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, bounds=(0, 1), num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer(dataset_type='IMAGENET', is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        print(acc) 

if __name__ == '__main__':
    unittest.main()
