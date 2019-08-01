from __future__ import absolute_import

import numpy as np
import warnings

from .base import Model


class PyTorchModel(Model):
    """Creates a :class:`Model` instance from a `PyTorch` module.
    Parameters
    ----------
    model : `torch.nn.Module`
        The PyTorch model that should be attacked.
    num_classes : int
        Number of classes for which the model will output predictions.
    channel_axis : int
        The index of the axis that represents color channels.
    device : string
        A string specifying the device to do computation on.
        If None, will default to "cuda:0" if torch.cuda.is_available()
        or "cpu" if not.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(
            self,
            model,
            num_classes,
            channel_axis=1,
            device=None,
            preprocessing=(0, 1)):

        # lazy import
        import torch

        super(PyTorchModel, self).__init__(channel_axis=channel_axis, preprocessing=preprocessing)

        self._num_classes = num_classes

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self._model = model.eval().to(self.device)

        if model.training:
            warnings.warn(
                'The PyTorch model is in training mode and therefore might'
                ' not be deterministic. Call the eval() method to set it in'
                ' evaluation mode if this is not intended.')

    def forward(self, inputs):
        # lazy import
        import torch

        inputs, _ = self._process_input(inputs)
        n = len(inputs)
        inputs = torch.from_numpy(inputs).to(self.device)

        predictions = self._model(inputs)
        predictions = predictions.detach().cpu().numpy()

        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        return predictions

    def num_classes(self):
        return self._num_classes

