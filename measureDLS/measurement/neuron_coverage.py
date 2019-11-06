import numpy as np

import measureDLS.utils as utils


class NeuronCoverage:

    def __init__(self, thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        self._thresholds = thresholds
        self._layer_neuron_id_to_global_neuron_id = {}
        self._results = {}
        self._num_neuron = 0
        self._num_input = 0

    def update(self, intermediate_layer_outputs, features_index):
        intermediate_layer_outputs_new = []
        for intermediate_layer_output in intermediate_layer_outputs:
            intermediate_layer_output = utils.to_numpy(intermediate_layer_output)
            intermediate_layer_outputs_new.append(intermediate_layer_output)
        intermediate_layer_outputs = intermediate_layer_outputs_new
        if len(self._results.keys()) == 0:
            current_global_neuron_id = 0
            for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                intermediate_layer_output_single_input = intermediate_layer_output[0]
                num_layer_neuron = intermediate_layer_output_single_input.shape[features_index]
                for layer_neuron_id in range(num_layer_neuron):
                    self._layer_neuron_id_to_global_neuron_id[(layer_id, layer_neuron_id)] = current_global_neuron_id
                    current_global_neuron_id += 1
                self._num_neuron += num_layer_neuron
            for threshold in self._thresholds:
                self._results[threshold] = np.zeros(shape=self._num_neuron)
        num_input = len(intermediate_layer_outputs[0])
        self._num_input += num_input
        for layer_id in range(len(intermediate_layer_outputs)):
            for input_id in range(num_input):
                intermediate_layer_outputs[layer_id][input_id] = (intermediate_layer_outputs[layer_id][input_id] - intermediate_layer_outputs[layer_id][input_id].min()) / (intermediate_layer_outputs[layer_id][input_id].max() - intermediate_layer_outputs[layer_id][input_id].min())
        for layer_id in range(len(intermediate_layer_outputs)):
            for input_id in range(num_input):
                num_layer_neuron = intermediate_layer_outputs[layer_id][input_id].shape[features_index]
                for layer_neuron_id in range(num_layer_neuron):
                    if features_index == -1:
                        neuron_output = intermediate_layer_outputs[layer_id][input_id][..., layer_neuron_id]
                    elif features_index == 0:
                        neuron_output = intermediate_layer_outputs[layer_id][input_id][layer_neuron_id]
                    else:
                        raise Exception('Invalid features index', features_index)
                    global_neuron_id = self._layer_neuron_id_to_global_neuron_id[(layer_id, layer_neuron_id)]
                    mean = np.mean(neuron_output)
                    for threshold in self._thresholds:
                        if mean > threshold:
                            self._results[threshold][global_neuron_id] = True

    def record(self, intermediate_layer_outputs, features_index):
        intermediate_layer_outputs_new = []
        for intermediate_layer_output in intermediate_layer_outputs:
            intermediate_layer_output = utils.to_numpy(intermediate_layer_output)
            intermediate_layer_outputs_new.append(intermediate_layer_output)
        intermediate_layer_outputs = intermediate_layer_outputs_new
        for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            np.save('/Users/zzh/Desktop/' + str(self._num_input) + '_' + str(layer_id), intermediate_layer_output)

    def report(self, *args):
        for threshold in self._thresholds:
            print('[NeuronCoverage] Time:{:s}, Num: {:d}, Threshold: {:.6f}, Neuron Coverage: {:.6f}({:d}/{:d})'.format(utils.readable_time_str(), self._num_input, threshold, self.get(threshold), len([v for v in self._results[threshold] if v]), self._num_neuron))

    def get(self, threshold):
        return len([v for v in self._results[threshold] if v]) / self._num_neuron if self._num_neuron != 0 else 0
