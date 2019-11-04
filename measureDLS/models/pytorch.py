from __future__ import absolute_import

import time

import foolbox
import torch

import measureDLS.utils as utils


class PyTorchModel:

    def __init__(self, model):
        assert isinstance(model, torch.nn.Module)
        self._model = model
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.eval()
        self._model.to(self._device)

    def predict(self, dataset, callbacks, batch_size=256):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self._device)
                labels = labels.to(self._device)
                y_mini_batch_pred = self._model(data)
                for callback in callbacks:
                    callback(labels, y_mini_batch_pred)

    def intermediate_layer_outputs(self, x, callbacks, batch_size=256):
        dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size)
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)
        for intermediate_layer in intermediate_layers:
            def hook(module, input, output):
                y_mini_batch_outputs.append(output)

            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)
        intermediate_layer_outputs_time = 0
        calc_time = 0
        with torch.no_grad():
            for data in dataloader:
                start = time.time()
                y_mini_batch_outputs.clear()
                data = data.to(self._device)
                self._model(data)
                end = time.time()
                intermediate_layer_outputs_time += end - start
                start = time.time()
                for callback in callbacks:
                    callback(y_mini_batch_outputs, 0)
                end = time.time()
                calc_time += end - start
        print('intermediate layer outputs time', intermediate_layer_outputs_time)
        print('calc time', calc_time)
        for handle in hook_handles:
            handle.remove()

    def adversarial_samples(self, dataset, bounds, num_classes, callbacks, batch_size=256, preprocessing=(0, 1), attack=foolbox.attacks.GradientAttack, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        foolbox_model = foolbox.models.PyTorchModel(self._model, bounds, num_classes, preprocessing=preprocessing, device=self._device)
        attack = attack(foolbox_model, criterion, distance, threshold)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for data, labels in dataloader:
            data = utils.to_numpy(data)
            labels = utils.to_numpy(labels)
            adversarials = attack(data, labels)
            for callback in callbacks:
                callback(data, adversarials)

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
            for intermediate_layer in intermediate_layers:
                print(intermediate_layer)
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
            print('$' * 50)
            for intermediate_layer in intermediate_layers:
                print(intermediate_layer)
        return intermediate_layers
