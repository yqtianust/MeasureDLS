import numpy as np

import measureDLS.utils as utils


class Accuracy:

    def __init__(self, k=1):
        self._k = k
        self._correct = 0
        self._total = 0

    def update(self, y_true, y_pred):
        y_true = utils.to_numpy(y_true)
        y_pred = utils.to_numpy(y_pred)
        size = len(y_true)
        top_k_predictions = np.argsort(y_pred)[:, -self._k:].T
        correct_matrix = np.zeros(size, bool)
        for i_th_prediction in top_k_predictions:
            correct_matrix = np.logical_or(correct_matrix, y_true == i_th_prediction)
        self._correct += len([v for v in correct_matrix if v])
        self._total += size

    @property
    def accuracy(self):
        if self._total == 0:
            return 0
        else:
            return self._correct / self._total
