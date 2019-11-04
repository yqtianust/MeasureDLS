import math

import foolbox


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
            if (origin == adversarial).all():
                continue
            self._num_total += 1
            if math.isnan(adversarial.max()):
                continue
            self._num_success += 1
            self._total_mean_squared_distance += foolbox.distances.MeanSquaredDistance(origin, adversarial, self._bounds).value
            self._total_mean_absolute_distance += foolbox.distances.MeanAbsoluteDistance(origin, adversarial, self._bounds).value
            self._total_linfinity += foolbox.distances.Linfinity(origin, adversarial, self._bounds).value
            self._total_l0 += foolbox.distances.L0(origin, adversarial, self._bounds).value

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
