from __future__ import absolute_import

import time
from collections import OrderedDict

import foolbox


class TensorFlowModel:

    def __init__(self, session, logits, input):
        self._session = session
        self._logits = logits
        self._input = input

    def predict(self, dataset, callbacks, batch_size=256):
        x = dataset[0]
        y = dataset[1]
        i = 0
        while i < len(x):
            data = x[i: i + batch_size]
            labels = y[i: i + batch_size]
            y_mini_batch_pred = self._session.run(self._logits, feed_dict={self._input: data})
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)
            i += batch_size

    def intermediate_layer_outputs(self, x, callbacks, batch_size=256):
        intermediate_layers = self._intermediate_layers()
        i = 0
        intermediate_layer_outputs_time = 0
        calc_time = 0
        while i < len(x):
            data = x[i: i + batch_size]
            start = time.time()
            y_mini_batch_outputs = self._session.run(intermediate_layers, feed_dict={self._input: data})
            end = time.time()
            intermediate_layer_outputs_time += end - start
            start = time.time()
            for callback in callbacks:
                callback(y_mini_batch_outputs, -1)
            end = time.time()
            calc_time += end - start
            i += batch_size
        print('intermediate layer outputs time', intermediate_layer_outputs_time)
        print('calc time', calc_time)

    def adversarial_samples(self, dataset, bounds, callbacks, batch_size=256, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        foolbox_model = foolbox.models.TensorFlowModel(self._input, self._logits, bounds, preprocessing=preprocessing)
        attack = attack(foolbox_model, criterion, distance, threshold)
        x = dataset[0]
        y = dataset[1]
        i = 0
        while i < len(x):
            data = x[i: i + batch_size]
            labels = y[i: i + batch_size]
            adversarials = attack(data, labels)
            for callback in callbacks:
                callback(data, adversarials)
            i += batch_size

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
        print(tensors)
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
        print(tensors)
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
