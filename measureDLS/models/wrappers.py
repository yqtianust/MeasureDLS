from __future__ import absolute_import

import numpy as np 
from .base import Model 

class ModelWrapper(Model):

    def __init__(self, model):
        super(ModelWrapper, self).__init__(bounds=model.bounds(), channel_axis=model.channel_axis())
        self.wrapped_model = model 

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def forward(self, inputs):
        return self.wrapped_model.forward(inputs)

    def num_classes(self):
        return self.wrapped_model.num_classes()

