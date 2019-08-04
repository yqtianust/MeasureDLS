from __future__ import absolute_import

import numpy as np
import logging

from .base import Model

class KerasModel(Model):
    """Creates a :class:`Model` instance from a `Keras` model.
    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first element of preprocessing
        from the input and then divide the input by the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    """

    def __init__(
            self,
            model,
            num_classes=None,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel, self).__init__()

        from keras import backend as K
        import keras
        from pkg_resources import parse_version

        assert parse_version(keras.__version__) >= parse_version(
            '2.0.7'), 'Keras version needs to be 2.0.7 or newer'

        self._num_classes = num_classes
        self._model = model 

    def forward(self, inputs):
        return self._model.predict(inputs)
        
    def evaluate(self, test_dataset):
        self._model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        test_datas, test_labels = test_dataset
        _, acc = self._model.evaluate(test_datas, test_labels)
        return acc

    def num_classes(self):
        return self._num_classes
