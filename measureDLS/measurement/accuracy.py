import numpy as np

import measureDLS.utils as utils


class Accuracy:

    def __init__(self, ks=(1, 5)):
        self._ks = ks
        self._correct = {}
        self._total = 0
        for k in self._ks:
            self._correct[k] = 0

    def update(self, y_true, y_pred):
        y_true = utils.to_numpy(y_true)
        y_pred = utils.to_numpy(y_pred)
        size = len(y_true)
        self._total += size
        for k in self._ks:
            top_k_predictions = np.argsort(y_pred)[:, -k:].T
            correct_matrix = np.zeros(size, bool)
            for i_th_prediction in top_k_predictions:
                correct_matrix = np.logical_or(correct_matrix, y_true == i_th_prediction)
            self._correct[k] += len([v for v in correct_matrix if v])

    def report(self, *args):
        for k in self._ks:
            print('[Accuracy] Time: {:s}, Num: {:d}, topK: {:d}, Accuracy: {:.6f}'.format(utils.readable_time_str(), self._total, k, self.get(k)))

    def get(self, k):
        if self._total == 0:
            return 0
        else:
            return self._correct[k] / self._total
