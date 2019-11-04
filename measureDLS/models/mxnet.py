from __future__ import absolute_import

import time

import foolbox
import mxnet

import measureDLS.utils as utils


class MXNetModel:

    def __init__(self, model):
        assert isinstance(model, mxnet.gluon.nn.Block)
        self._model = model
        self._best_ctx = mxnet.gpu() if mxnet.test_utils.list_gpus() else mxnet.cpu()
        self._model.collect_params().reset_ctx(self._best_ctx)

    def predict(self, dataset, callbacks, batch_size=256):
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = data.as_in_context(self._best_ctx)
            labels = labels.as_in_context(self._best_ctx)
            y_mini_batch_pred = self._model(data)
            for callback in callbacks:
                callback(labels, y_mini_batch_pred)

    def intermediate_layer_outputs(self, x, callbacks, batch_size=256):
        dataloader = mxnet.gluon.data.DataLoader(x, batch_size=batch_size)
        inputs = mxnet.sym.var('data')
        outputs = self._intermediate_layer_names()
        feat_model = mxnet.gluon.SymbolBlock(outputs, inputs, params=self._model.collect_params())
        intermediate_layer_outputs_time = 0
        wait_time = 0
        calc_time = 0
        for data in dataloader:
            start = time.time()
            data = data.as_in_context(self._best_ctx)
            y_mini_batch_outputs = feat_model(data)
            end = time.time()
            intermediate_layer_outputs_time += end - start
            start = time.time()
            for y_mini_batch_output in y_mini_batch_outputs:
                y_mini_batch_output.wait_to_read()
            end = time.time()
            wait_time += end - start
            start = time.time()
            for callback in callbacks:
                callback(y_mini_batch_outputs, 0)
            end = time.time()
            calc_time += end - start
        print('intermediate layer outputs time', intermediate_layer_outputs_time)
        print('wait time', wait_time)
        print('calc time', calc_time)

    def adversarial_samples(self, dataset, bounds, num_classes, callbacks, batch_size=256, preprocessing=(0, 1), attack=foolbox.attacks.GradientAttack, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        foolbox_model = foolbox.models.MXNetGluonModel(self._model, bounds, num_classes, preprocessing=preprocessing, ctx=self._best_ctx)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = mxnet.gluon.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = utils.to_numpy(data)
            labels = utils.to_numpy(labels)
            adversarials = attack(data, labels)
            for callback in callbacks:
                callback(data, adversarials)

    def _intermediate_layer_names(self):
        inputs = mxnet.sym.var('data')
        internals = self._model(inputs).get_internals()
        outputs = []
        for internal in internals:
            name = internal.name.lower()
            print(name)
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
        print('$' * 50)
        print(outputs)
        return outputs
