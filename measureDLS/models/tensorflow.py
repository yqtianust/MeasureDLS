from __future__ import absolute_import

import warnings
from collections import OrderedDict

import foolbox
import numpy as np
import tensorflow as tf


class TensorFlowModel:

    def __init__(self, session, logits, input):
        self._session = session
        self._logits = logits
        self._input = input

    def predict(self, x, y, callbacks, batch_size=64):
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            labels = y[index:index + batch_size]
            y_mini_batch_pred = self._session.run(self._logits, feed_dict={self._input: data})
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)
            index += batch_size

    def intermediate_layer_outputs(self, x, callbacks, batch_size=32):
        intermediate_layers = self._intermediate_layers()
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            y_mini_batch_outputs = self._session.run(intermediate_layers, feed_dict={self._input: data})
            for callback in callbacks:
                callback(y_mini_batch_outputs, -1)
            index += batch_size

    def adversarial_samples(self, x, y, num_tries, bounds, callbacks, batch_size=16, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            tf.get_logger().setLevel('ERROR')
            foolbox_model = foolbox.models.TensorFlowModel(self._input, self._logits, bounds, preprocessing=preprocessing)
            tf.get_logger().setLevel('DEBUG')
        attack = attack(foolbox_model, criterion, distance, threshold)
        index = 0
        while index < len(x):
            data = x[index:index + batch_size]
            labels = y[index:index + batch_size]
            index += batch_size
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                adversarials = attack(data, labels)
            tries = []
            for i in range(data.shape[0]):
                if (data[i] == adversarials[i]).all():
                    continue
                tries.append(i)
            if len(tries) == 0:
                continue
            if num_tries < len(tries):
                tries = tries[:num_tries]
            data_filtered = np.empty(shape=((len(tries),) + data.shape[1:]))
            adversarials_filtered = np.empty(shape=((len(tries),) + adversarials.shape[1:]))
            for i in range(len(tries)):
                data_filtered[i] = data[tries[i]]
                adversarials_filtered[i] = adversarials[tries[i]]
            data = data_filtered
            adversarials = adversarials_filtered
            for callback in callbacks:
                callback(data, adversarials)
            num_tries -= len(tries)
            if num_tries == 0:
                break

    def _intermediate_layers(self):
        ordered_tensors = self._ordered_tensors_in_graph(self._logits)
        tensors = []
        for tensor in ordered_tensors:
            if not len(tensor.shape) > 0:
                continue
            if tensor.shape[0] is not None and str(tensor.shape[0]) != '?':
                continue
            if not len(tensor.op.inputs._inputs) > 0:
                continue
            if 'Reshape' == tensor.op.type:
                continue
            if 'Squeeze' == tensor.op.type:
                continue
            if 'Pool' in tensor.op.type:
                continue
            if 'MatMul' in tensor.op.type:
                continue
            if 'Identity' in tensor.op.type:
                continue
            if 'Pad' in tensor.op.type:
                continue
            tensors.append(tensor)
        filtered_tensors = []
        for i in range(len(tensors)):
            if i == len(tensors) - 1:
                filtered_tensors.append(tensors[i])
                continue
            if 'Relu' in tensors[i].op.type:
                filtered_tensors.append(tensors[i])
                continue
            if 'Relu' in tensors[i + 1].op.type:
                continue
            if 'Conv' in tensors[i].op.type and 'Add' in tensors[i + 1].op.type:
                continue
            if 'Conv' in tensors[i].op.type and 'BatchNorm' in tensors[i + 1].op.type:
                continue
            filtered_tensors.append(tensors[i])
        tensors = filtered_tensors
        return tensors

    def _ordered_tensors_in_graph(self, tensor, entrypoint=True):
        tensors_in_order = [tensor]
        for input_tensor in tensor.op.inputs._inputs:
            next_tensors_in_order = self._ordered_tensors_in_graph(input_tensor, False)
            tensors_in_order += next_tensors_in_order
        if entrypoint:
            tensors_in_order = list(OrderedDict.fromkeys(tensors_in_order))
            list.reverse(tensors_in_order)
        return tensors_in_order
