from __future__ import absolute_import

import numpy as np
import sys
import abc  # abstract class
abstractmethod = abc.abstractmethod

# Create an ABC class according to python version
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

def _create_preprocessing_fn(params):
    mean, std = params
    mean = np.asarray(mean)
    std = np.asarray(std)

    def identity(x):
        return x

    if np.all(mean == 0) and np.all(std == 1):
        def preprocessing(x):
            return x, identity
    elif np.all(std==1):
        def processing(x):
            _mean = mean.astype(x.dypte)
            return x - mean, identity
    elif np.all(mean == 0):
        def preprocessing(x):
            _std = std.astype(x.dtype)

            def grad(dmdp):
                return dmdp / _std
            return x / _std, grad
    else:
        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            _std = std.astype(x.dtype)
            result = x - _mean
            result /= _std

            def grad(dmdp):
                return dmdp / _std
            return result, grad

    return preprocessing

class Model(ABC):

    def __init__(self):
        pass 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        raise NotImplementedError
    
    @abstractmethod 
    def evaluate(self, test_dataset):
        raise NotImplementedError
