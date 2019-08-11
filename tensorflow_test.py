import warnings
import unittest

import measureDLS
from measureDLS.dataset_prepare import prepare_imagenet_val_dataset 

import numpy as np
import tensorflow as tf 

class TestTensorflow(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        '''
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)
        '''

    def test_pretrained_model_Tensorflow_model1_IMAGENET_accurancy(self):
        user_model = tf.keras.applications.MobileNetV2(weights='imagenet')
        wrapped_model = measureDLS.models.TensorFlowModel(user_model, num_classes=1000)
        accurancy_measurer = measureDLS.measurement.AccurancyMeasurer('IMAGENET', is_input_flatten=False, preprocess=None)
        acc = accurancy_measurer.measure_accurancy(wrapped_model)
        print(acc)
        # self.assertGreaterEqual(acc, 0.713 - 0.05) # 0.707
    
    def test_pretrained_model_Tensorflow_model2_IMAGENET_accurancy(self):
        pass 

    def test_pretrained_model_Tensorflow_model3_IMAGENET_accurancy(self):
        pass 


if __name__ == '__main__':
    unittest.main()
