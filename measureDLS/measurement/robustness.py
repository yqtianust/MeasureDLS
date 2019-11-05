import math

import foolbox

from measureDLS.utils import utils


class Robustness:

    def __init__(self, bounds):
        self._bounds = bounds
        self._num_total = 0
        self._num_success = 0
        self._total_mean_squared_distance = 0
        self._total_mean_absolute_distance = 0
        self._total_linfinity = 0
        self._total_l0 = 0

    def update(self, origins, adversarials):
        for i in range(len(origins)):
            origin = origins[i]
            adversarial = adversarials[i]
            self._num_total += 1
            if math.isnan(adversarial.max()):
                continue
            self._num_success += 1
            self._total_mean_squared_distance += foolbox.distances.MeanSquaredDistance(origin, adversarial, self._bounds).value
            self._total_mean_absolute_distance += foolbox.distances.MeanAbsoluteDistance(origin, adversarial, self._bounds).value
            self._total_linfinity += foolbox.distances.Linfinity(origin, adversarial, self._bounds).value
            self._total_l0 += foolbox.distances.L0(origin, adversarial, self._bounds).value

    def report(self, *args):
        print('[Robustness] Time: {:s}, Success Rate: {:.10f}({:d}/{:d}), Avg Mean Squared Distance: {:.10f}, Avg Mean Absolute Distance: {:.10f}, Avg Linfinity: {:.10f}, Avg L0: {:.10f}'.format(utils.readable_time_str(), self.success_rate, self._num_success, self._num_total, self.avg_mean_squared_distance, self.avg_mean_absolute_distance, self.avg_linfinity, self.avg_l0))

    @property
    def success_rate(self):
        return self._num_success / self._num_total

    @property
    def avg_mean_squared_distance(self):
        return self._total_mean_squared_distance / self._num_total

    @property
    def avg_mean_absolute_distance(self):
        return self._total_mean_absolute_distance / self._num_total

    @property
    def avg_linfinity(self):
        return self._total_linfinity / self._num_total

    @property
    def avg_l0(self):
        return self._total_l0 / self._num_total
