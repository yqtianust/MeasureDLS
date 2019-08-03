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
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)

    def test_pretrained_model_Keras_Vgg15_IMAGENET_accurancy(self):
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input

        user_model = VGG16(weights='imagenet', include_top=False)
        wrapped_model = measureDLS.models.KerasModel(user_model num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        print(acc)

if __name__ == '__main__':
    unittest.main()

