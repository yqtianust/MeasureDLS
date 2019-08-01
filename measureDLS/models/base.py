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

    def __init__(self, channel_axis, preprocessing=(0, 1)):
        self._channel_axis = channel_axis

        if not callable(preprocessing):
            preprocessing = _create_preprocessing_fn(preprocessing)
        assert callable(preprocessing)
        self._preprocessing = preprocessing

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def channel_axis(self):
        return self._channel_axis

    def _process_input(self, x):
        p, grad = self._preprocessing(x)
        if hasattr(p, 'dtype'):
            assert p.dtype == x.dtype
        p = np.asarray(p, dtype=x.dtype)
        assert callable(grad)
        return p, grad

    def _process_gradient(self, backward, dmdp):
        if backward is None:
            raise ValueError('Your preprocessing function does not provide an (approximate) gradient')
        
        dmdx = backward(dmdp)
        assert dmdx.dtype == dmdp.dtype
        return dmdx

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    def forward_one(self, x):
        return np.squeeze(self.forward(x[np.newaxis]), axis=0)

    @abstractmethod
    def num_classes(self):
        raise NotImplementedError


