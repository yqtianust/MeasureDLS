from __future__ import absolute_import

import numpy as np
import logging

from .base import Model


class TensorFlowModel(Model):
    """Creates a :class:`Model` instance from existing `TensorFlow` tensors.
    Parameters
    ----------
    inputs : `tensorflow.Tensor`
        The input to the model, usually a `tensorflow.placeholder`.
    logits : `tensorflow.Tensor`
        The predictions of the model, before the softmax.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model, 
            num_classes=None, 
            channel_axis=3,
            preprocessing=(0, 1)):

        super(TensorFlowModel, self).__init__()

        # delay import until class is instantiated
        import tensorflow as tf

        self._model = model
        self._num_classes = num_classes

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def num_classes(self):
        return self._num_classes

    def forward(self, inputs):
        # Fix the process issue here 
        inputs, _ = self._process_input(inputs)
        predictions = self._session.run(self._logits, feed_dict={self._inputs: inputs})
        return predictions

    def evaluate(self, test_dataset):
        # Wait for integrate after testing in a Linux environment
        self._model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        test_datas, test_labels = test_dataset
        _, acc = self._model.evaluate(test_datas, test_labels)
        return acc


