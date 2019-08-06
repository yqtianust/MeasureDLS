import warnings
import unittest

import torch 
from torchvision import models
from torchvision import transforms

import measureDLS

MNIST_TRANSFORM = transforms.ToTensor()
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

class TestPyTorch(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)

    def test_sample_model_Pytorch_FC_MNIST_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_FC_MNIST()    
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('MNIST', MNIST_TRANSFORM, is_input_flatten=True)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.91)

    def test_sample_model_Pytorch_CNN_MNIST_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_CNN_MNIST()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('MNIST', MNIST_TRANSFORM, is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.97)


    def test_sample_model_Pytorch_FC_CIFAR10_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_FC_CIFAR10()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('CIFAR10', CIFAR10_TRANSFORM, is_input_flatten=True)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.47)

    def test_sample_model_Pytorch_CNN_CIFAR10_accurancy(self):
        user_model = measureDLS.utils.load_sample_Pytroch_CNN_CIFAR10()
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=10)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('CIFAR10', CIFAR10_TRANSFORM, is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.6)

    def test_pretrained_model_Pytorch_Alexnet_IMAGENET_accurancy(self):
        user_model = models.alexnet(pretrained=True)
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', IMAGENET_TRANSFORM, is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.75)

    def test_pretrained_model_Pytorch_Vgg11_IMAGENET_accurancy(self):
        user_model = models.vgg11(pretrained=True)
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', IMAGENET_TRANSFORM, is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.85)

    def test_pretrained_model_Pytorch_Resnet_IMAGENET_accurancy(self):
        user_model = models.resnet34(pretrained=True)
        wrapped_model = measureDLS.models.PyTorchModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', IMAGENET_TRANSFORM, is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.88) 

if __name__ == '__main__':
    unittest.main()
