from __future__ import absolute_import

import warnings

import foolbox
import numpy as np
import torch

import measureDLS.utils as utils


class PyTorchModel:

    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self._model = model
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.eval()
        self._model.to(self._device)

    def predict(self, dataset, callbacks, batch_size=64):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self._device)
                labels = labels.to(self._device)
                y_mini_batch_pred = self._model(data)
                for callback in callbacks:
                    callback(labels, y_mini_batch_pred)

    def intermediate_layer_outputs(self, dataset, callbacks, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        for intermediate_layer in intermediate_layers:
            def hook(module, input, output):
                y_mini_batch_outputs.append(output)

            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)
        with torch.no_grad():
            for data in dataloader:
                if isinstance(data, list):
                    data = data[0]
                y_mini_batch_outputs.clear()
                data = data.to(self._device)
                self._model(data)
                for callback in callbacks:
                    callback(y_mini_batch_outputs, 0)
        for handle in hook_handles:
            handle.remove()

    def adversarial_samples(self, dataset, num_tries, bounds, num_classes, callbacks, batch_size=16, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            foolbox_model = foolbox.models.PyTorchModel(self._model, bounds, num_classes, preprocessing=preprocessing, device=self._device)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
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

    def _intermediate_layers(self, module, entrypoint=True):
        intermediate_layers = []
        for submodule in module.children():
            if len(submodule._modules) > 0:
                intermediate_layers += self._intermediate_layers(submodule, False)
            else:
                if 'Pool' in str(submodule.type):
                    continue
                if 'Dropout' in str(submodule.type):
                    continue
                intermediate_layers.append(submodule)
        if entrypoint:
            filtered_intermediate_layers = []
            for i in range(len(intermediate_layers)):
                if i == len(intermediate_layers) - 1:
                    filtered_intermediate_layers.append(intermediate_layers[i])
                    continue
                next_intermediate_layer_type = str(intermediate_layers[i + 1].type).lower()
                if 'batchnorm' in next_intermediate_layer_type:
                    continue
                if 'relu' in next_intermediate_layer_type:
                    continue
                filtered_intermediate_layers.append(intermediate_layers[i])
            intermediate_layers = filtered_intermediate_layers
        return intermediate_layers
