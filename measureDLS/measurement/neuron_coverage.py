import math
import multiprocessing

import numpy as np

import measureDLS.utils as utils


class NeuronCoverage:

    def __init__(self, thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        self._thresholds = thresholds
        self._results = {}
        self._num_neuron = 0
        self._layer_neuron_start_id_list = []
        self._neuron_id_to_layer_id_dict = {}
        self._features_index = None
        self._num_input = 0

    def update(self, intermediate_layer_outputs, features_index):
        intermediate_layer_outputs_new = []
        for intermediate_layer_output in intermediate_layer_outputs:
            intermediate_layer_output = utils.to_numpy(intermediate_layer_output)
            intermediate_layer_output = (intermediate_layer_output - intermediate_layer_output.min()) / (intermediate_layer_output.max() - intermediate_layer_output.min())
            intermediate_layer_outputs_new.append(intermediate_layer_output)
        intermediate_layer_outputs = intermediate_layer_outputs_new
        if len(self._results.keys()) == 0:
            for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                for input_id, intermediate_layer_output_single_input in enumerate(intermediate_layer_output):
                    if input_id == 0:
                        self._layer_neuron_start_id_list.append(self._num_neuron)
                        self._num_neuron += intermediate_layer_output_single_input.shape[features_index]
            self._layer_neuron_start_id_list.append(self._num_neuron)
            for threshold in self._thresholds:
                self._results[threshold] = np.zeros(shape=self._num_neuron)
            self._features_index = features_index
            layer_id = 0
            for neuron_id in range(self._num_neuron):
                if not (self._layer_neuron_start_id_list[layer_id] <= neuron_id < self._layer_neuron_start_id_list[layer_id + 1]):
                    layer_id += 1
                self._neuron_id_to_layer_id_dict[neuron_id] = layer_id
        num_input = len(intermediate_layer_outputs[0])
        self._num_input += num_input
        total = self._num_neuron * num_input
        cpu_count = multiprocessing.cpu_count()
        num_each_process = math.ceil(total / cpu_count)
        processes = []
        queue = multiprocessing.Queue()
        for i in range(cpu_count):
            process = multiprocessing.Process(target=self._update_worker,
                                              args=(intermediate_layer_outputs,
                                                    i * num_each_process,
                                                    (i + 1) * num_each_process if (i + 1) * num_each_process <= total else total,
                                                    queue))
            processes.append(process)
            process.start()
        for _ in processes:
            results_worker = queue.get()
            for threshold in self._thresholds:
                self._results[threshold] = np.logical_or(self._results[threshold], results_worker[threshold])
        for process in processes:
            process.join()

    def report(self, *args):
        for threshold in self._thresholds:
            print('[NeuronCoverage] Time: {:s}, Num: {:d}, Threshold: {:.6f}, Neuron Coverage: {:.6f}({:d}/{:d})'.format(utils.readable_time_str(), self._num_input, threshold, self.get(threshold), len([v for v in self._results[threshold] if v]), self._num_neuron))

    def get(self, threshold):
        return len([v for v in self._results[threshold] if v]) / self._num_neuron if self._num_neuron != 0 else 0

    def _update_worker(self, intermediate_layer_outputs, start, end, queue):
        for i in range(start, end):
            input_id = i // self._num_neuron
            neuron_id = i % self._num_neuron
            need_calc = False
            for threshold in self._thresholds:
                if not self._results[threshold][neuron_id]:
                    need_calc = True
                    break
            if not need_calc:
                continue
            layer_id = self._neuron_id_to_layer_id_dict[neuron_id]
            neuron_id_in_this_layer = neuron_id - self._layer_neuron_start_id_list[layer_id]
            intermediate_layer_output_single_input = intermediate_layer_outputs[layer_id][input_id]
            if self._features_index == -1:
                neuron_output = intermediate_layer_output_single_input[..., neuron_id_in_this_layer]
            elif self._features_index == 0:
                neuron_output = intermediate_layer_output_single_input[neuron_id_in_this_layer]
            else:
                raise Exception('Invalid features index', self._features_index)
            mean = np.mean(neuron_output)
            for threshold in self._thresholds:
                if mean > threshold:
                    self._results[threshold][neuron_id] = True
        queue.put(self._results)
