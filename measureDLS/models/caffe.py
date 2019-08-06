from __future__ import absolute_import

import numpy as np
import warnings
from .base import Model 
from .. import utils


class CaffeModel(Model):  # pragma: no cover
    def __init__(self,
                 net,
                 bounds,
                 channel_axis=1,
                 preprocessing=(0, 1),
                 data_blob_name="data",
                 label_blob_name="label",
                 output_blob_name="output"):
        super(CaffeModel, self).__init__()

        warnings.warn('Caffe was superseeded by Caffe2 and now PyTorch 1.0,'
                      ' thus Caffe support in Foolbox will be removed',
                      DeprecationWarning)

        import caffe
        self.net = net
        assert isinstance(net, caffe.Net)
        assert data_blob_name in self.net.blobs
        assert label_blob_name in self.net.blobs
        self.data_blob_name = data_blob_name
        self.label_blob_name = label_blob_name
        self.output_blob_name = output_blob_name

    def num_classes(self):
        return self.net.blobs[self.output_blob_name].data.shape[-1]

    def forward(self, inputs):
        inputs, _ = self._process_input(inputs)
        self.net.blobs[self.data_blob_name].reshape(*inputs.shape)
        self.net.blobs[self.label_blob_name].reshape(inputs.shape[0])
        self.net.blobs[self.data_blob_name].data[:] = inputs
        self.net.forward()
        return self.net.blobs[self.output_blob_name].data

    def _loss_fn(self, x, label):
        logits = self.forward(x[None])
        return utils.batch_crossentropy([label], logits)

    def num_classes(self):
        return self._num_classes

    def evaluate(self):
        pass 
