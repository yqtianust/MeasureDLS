from __future__ import absolute_import

import warnings

import foolbox
import mxnet
import numpy as np

import measureDLS.utils as utils


class MXNetModel:

    def __init__(self, model):
        assert isinstance(model, mxnet.gluon.nn.Block)
        self._model = model
        self._best_ctx = mxnet.gpu() if mxnet.test_utils.list_gpus() else mxnet.cpu()
        self._model.collect_params().reset_ctx(self._best_ctx)

    def predict(self, dataset, callbacks, batch_size=64):
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = data.as_in_context(self._best_ctx)
            labels = labels.as_in_context(self._best_ctx)
            y_mini_batch_pred = self._model(data)
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)

    def intermediate_layer_outputs(self, dataset, callbacks, batch_size=32):
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        inputs = mxnet.sym.var('data')
        outputs = self._intermediate_layer_names()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            feat_model = mxnet.gluon.SymbolBlock(outputs, inputs, params=self._model.collect_params())
        for data in dataloader:
            if isinstance(data, list):
                data = data[0]
            data = data.as_in_context(self._best_ctx)
            y_mini_batch_outputs = feat_model(data)
            for y_mini_batch_output in y_mini_batch_outputs:
                y_mini_batch_output.wait_to_read()
            for callback in callbacks:
                callback(y_mini_batch_outputs, 0)

    def adversarial_samples(self, dataset, num_tries, bounds, num_classes, callbacks, batch_size=16, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            foolbox_model = foolbox.models.MXNetGluonModel(self._model, bounds, num_classes, preprocessing=preprocessing, ctx=self._best_ctx)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = utils.to_numpy(data)
            labels = utils.to_numpy(labels)
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

    def _intermediate_layer_names(self):
        inputs = mxnet.sym.var('data')
        internals = self._model(inputs).get_internals()
        outputs = []
        for internal in internals:
            name = internal.name.lower()
            if ('_fwd' not in name) and ('relu' not in name) and ('activation' not in name):
                continue
            if 'flatten' in name:
                continue
            if 'reshape' in name:
                continue
            if 'dropout' in name:
                continue
            if 'pool' in name:
                continue
            if 'input' in name:
                continue
            if 'pad' in name:
                continue
            outputs.append(internal)
        filtered_outputs = []
        for i in range(len(outputs)):
            if i == len(outputs) - 1:
                filtered_outputs.append(outputs[i])
                continue
            if 'relu' in outputs[i].name or 'activation' in outputs[i].name:
                filtered_outputs.append(outputs[i])
                continue
            if 'relu' in outputs[i + 1].name or 'activation' in outputs[i + 1].name:
                continue
            if 'batchnorm' in outputs[i + 1].name:
                continue
            filtered_outputs.append(outputs[i])
        outputs = filtered_outputs
        return outputs
