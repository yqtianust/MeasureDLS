import math
import multiprocessing

import numpy as np

import measureDLS.utils as utils


class NeuronCoverage:

    def __init__(self, threshold=0.6):
        self._threshold = threshold
        self._result = None
        self._num_neuron = 0
        self._layer_neuron_start_id_list = []
        self._neuron_id_to_layer_id_dict = {}
        self._features_index = None

    def update(self, intermediate_layer_outputs, features_index):
        intermediate_layer_outputs_new = []
        for intermediate_layer_output in intermediate_layer_outputs:
            intermediate_layer_output = utils.to_numpy(intermediate_layer_output)
            intermediate_layer_output = (intermediate_layer_output - intermediate_layer_output.min()) / (intermediate_layer_output.max() - intermediate_layer_output.min())
            intermediate_layer_outputs_new.append(intermediate_layer_output)
        intermediate_layer_outputs = intermediate_layer_outputs_new
        if self._result is None:
            for layer_id, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                print(intermediate_layer_output.shape)
                for input_id, intermediate_layer_output_single_input in enumerate(intermediate_layer_output):
                    if input_id == 0:
                        self._layer_neuron_start_id_list.append(self._num_neuron)
                        self._num_neuron += intermediate_layer_output_single_input.shape[features_index]
            self._layer_neuron_start_id_list.append(self._num_neuron)
            self._result = np.zeros(shape=self._num_neuron)
            self._features_index = features_index
            layer_id = 0
            for neuron_id in range(self._num_neuron):
                if not (self._layer_neuron_start_id_list[layer_id] <= neuron_id < self._layer_neuron_start_id_list[layer_id + 1]):
                    layer_id += 1
                self._neuron_id_to_layer_id_dict[neuron_id] = layer_id
        total = self._num_neuron * len(intermediate_layer_outputs[0])
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
            self._result = np.logical_or(self._result, queue.get())
        for process in processes:
            process.join()

    @property
    def neuron_covered(self):
        return len([v for v in self._result if v])

    @property
    def neuron_total(self):
        return self._num_neuron

    @property
    def neuron_coverage(self):
        return self.neuron_covered / self.neuron_total if self.neuron_total != 0 else 0

    def _update_worker(self, intermediate_layer_outputs, start, end, queue):
        for i in range(start, end):
            input_id = i // self._num_neuron
            neuron_id = i % self._num_neuron
            if self._result[neuron_id] == True:
                continue
            layer_id = self._neuron_id_to_layer_id_dict[neuron_id]
            neuron_id_in_this_layer = neuron_id - self._layer_neuron_start_id_list[layer_id]
            intermediate_layer_output_single_input = intermediate_layer_outputs[layer_id][input_id]
            if self._features_index == -1:
                neuron_output = intermediate_layer_output_single_input[..., neuron_id_in_this_layer]
            elif self._features_index == 0:
                neuron_output = intermediate_layer_output_single_input[neuron_id_in_this_layer]
            else:
                raise Exception("Invalid features index", self._features_index)
            if np.mean(neuron_output) > self._threshold:
                self._result[neuron_id] = True
        queue.put(self._result)
