import numpy as np


class RegressionModel:
    _x_train = None
    _y_train = None
    _x_validation = None
    _y_validation = None

    def __inti__(self, x_train=None, y_train=None, x_validation=None, y_validation=None):
        self._x_train = x_train
        self._y_train = y_train
        self._x_validation = x_validation
        self._y_validation = y_validation

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        pass

    def evaluate_model(self):
        pass

    def get_model(self):
        pass

    def set_model(self):
        pass

    # get the name of the model as string
    def to_string(self):
        pass
