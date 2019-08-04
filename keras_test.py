import warnings
import unittest

import measureDLS
import keras 

from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img

import numpy as np
# vgg_16, vgg_19 and resnet all accept 224x224 input images
# inception_v3 and Xception require 299*299 pixel inputs 

class TestMeasureDLS(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        '''
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)
        '''

    def test_pretrained_model_Keras_Vgg16_IMAGENET_accurancy(self):
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        
        user_model = VGG16(weights='imagenet', include_top=True)
        wrapped_model = measureDLS.models.KerasModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=preprocess_input)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.713 - 0.05) # 0.707

    def test_pretrained_model_Keras_ResNet50_IMAGENET_accurancy(self):
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input

        user_model = ResNet50(weights='imagenet')
        wrapped_model = measureDLS.models.KerasModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=preprocess_input)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.749 - 0.05) # 0.756

    def test_pretrained_model_Keras_MobileNet_IMAGENET_accurancy(self):
        from keras.applications.mobilenet import MobileNet 
        from keras.applications.mobilenet import preprocess_input 

        user_model = MobileNet(weights='imagenet')
        wrapped_model = measureDLS.models.KerasModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=preprocess_input)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        self.assertGreaterEqual(acc, 0.704 - 0.05) # 0.709

if __name__ == '__main__':
    unittest.main()

