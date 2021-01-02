from regression_model import RegressionModel
from sklearn.svm import SVR


class SupportVectorRegression(RegressionModel):

    _scaling_x = True
    _scaling_y = True

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, scaling_x=True,
                 scaling_y=True):
        self._scaling_x = scaling_x
        self._scaling_y = scaling_y

        if model is None:
            model = SVR(kernel="rbf")
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        pass

    def evaluate_model(self):
        pass

    # get the name of the model as string
    def to_string(self):
        pass
