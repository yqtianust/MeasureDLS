import warnings
import unittest

import measureDLS

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

if __name__ == '__main__':
    unittest.main()
