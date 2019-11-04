from __future__ import absolute_import

import time

import foolbox
import tensorflow as tf


class TensorFlowEagerModel:

    def __init__(self, model):
        assert isinstance(model, tf.keras.Model)
        self._model = model

    def predict(self, dataset, callbacks, batch_size=256):
        x = dataset[0]
        y = dataset[1]

        class Callback(tf.keras.callbacks.Callback):
            def on_predict_batch_end(self, batch, logs=None):
                y_mini_batch_pred = logs['outputs'][0]
                labels = y[batch * batch_size: (batch + 1) * batch_size]
                for callback in callbacks:
                    callback(labels, y_mini_batch_pred)

        self._model.predict(x, batch_size=batch_size, callbacks=[Callback()])

    def intermediate_layer_outputs(self, x, callbacks, batch_size=256):
        layer_names = self._intermediate_layer_names()
        intermediate_layer_model = tf.keras.Model(inputs=self._model.input, outputs=[self._model.get_layer(layer_name).output for layer_name in layer_names])
        intermediate_layer_outputs_time = 0
        calc_time = 0
        start = end = time.time()

        class Callback(tf.keras.callbacks.Callback):
            def on_predict_batch_end(self, batch, logs=None):
                nonlocal intermediate_layer_outputs_time, calc_time, start, end
                end = time.time()
                intermediate_layer_outputs_time += end - start
                start = time.time()
                y_mini_batch_outputs = logs['outputs']
                for callback in callbacks:
                    callback(y_mini_batch_outputs, -1)
                end = time.time()
                calc_time += end - start
                start = time.time()

        intermediate_layer_model.predict(x, batch_size=batch_size, callbacks=[Callback()])
        print('intermediate layer outputs time', intermediate_layer_outputs_time)
        print('calc time', calc_time)

    def adversarial_samples(self, dataset, bounds, callbacks, batch_size=256, preprocessing=(0, 1), attack=foolbox.attacks.GradientAttack, criterion=foolbox.criteria.Misclassification(), distance=foolbox.distances.MSE, threshold=None):
        foolbox_model = foolbox.models.TensorFlowEagerModel(self._model, bounds, preprocessing=preprocessing)
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

    def _intermediate_layer_names(self):
        layer_names = []
        for layer in self._model.layers:
            print(layer.name)
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
        print('$' * 50)
        print(layer_names)
        return layer_names
