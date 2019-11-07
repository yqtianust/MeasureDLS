from __future__ import absolute_import

import math
import warnings

import foolbox
import numpy as np
import tensorflow as tf


class TensorFlowEagerModel:

    def __init__(self, model):
        assert isinstance(model, tf.keras.Model)
        self._model = model

    def predict(self, x, y, callbacks, batch_size=64):
        generator = SimpleDataGenerator(x, y, batch_size)
        return self.predict_generator(generator, callbacks)

    def predict_generator(self, generator, callbacks):
        class Callback(tf.keras.callbacks.Callback):
            def on_predict_batch_end(self, batch, logs=None):
                y_mini_batch_pred = logs['outputs'][0]
                labels = generator[batch][1]
                for callback in callbacks:
                    callback(labels, y_mini_batch_pred)

        self._model.predict_generator(generator, callbacks=[Callback()])

    def intermediate_layer_outputs(self, x, callbacks, batch_size=32):
        generator = SimpleDataGenerator(x, None, batch_size)
        return self.intermediate_layer_outputs_generator(generator, callbacks)

    def intermediate_layer_outputs_generator(self, generator, callbacks):
        class Callback(tf.keras.callbacks.Callback):
            def on_predict_batch_end(self, batch, logs=None):
                y_mini_batch_outputs = logs['outputs']
                for callback in callbacks:
                    callback(y_mini_batch_outputs, -1)

        layer_names = self._intermediate_layer_names()
        intermediate_layer_model = tf.keras.Model(inputs=self._model.input, outputs=[self._model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_model.predict_generator(generator, callbacks=[Callback()])

    def adversarial_samples(self, x, y, num_tries, bounds, callbacks, batch_size=16, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        generator = SimpleDataGenerator(x, y, batch_size)
        return self.adversarial_samples_generator(generator, num_tries, bounds, callbacks, preprocessing=preprocessing, attack=attack, criterion=criterion, distance=distance, threshold=threshold)

    def adversarial_samples_generator(self, generator, num_tries, bounds, callbacks, preprocessing=(0, 1), attack=foolbox.attacks.FGSM, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            foolbox_model = foolbox.models.TensorFlowEagerModel(self._model, bounds, preprocessing=preprocessing)
        attack = attack(foolbox_model, criterion, distance, threshold)
        for i in range(len(generator)):
            data, labels = generator[i]
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
        layer_names = []
        for layer in self._model.layers:
            name = layer.name.lower()
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
            layer_names.append(layer.name)
        filtered_layer_names = []
        for i in range(len(layer_names)):
            if i == len(layer_names) - 1:
                filtered_layer_names.append(layer_names[i])
                continue
            name = layer_names[i].lower()
            next_name = layer_names[i + 1].lower()
            if 'relu' in name or 'activation' in name:
                filtered_layer_names.append(layer_names[i])
                continue
            if 'relu' in next_name or 'activation' in next_name:
                continue
            if '_bn' in next_name or 'bn_' in next_name:
                continue
            filtered_layer_names.append(layer_names[i])
        layer_names = filtered_layer_names
        return layer_names


class SimpleDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x, y, batch_size):
        self._x = x
        self._y = y
        self._batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self._x) / self._batch_size)

    def __getitem__(self, index):
        start = index * self._batch_size
        end = (index + 1) * self._batch_size
        x = self._x[start:end]
        if self._y is not None:
            y = self._y[start:end]
            return x, y
        else:
            return x, None
