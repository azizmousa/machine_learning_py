from regression_model import RegressionModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SupportVectorRegression(RegressionModel):

    _x_scaler = None
    _y_scaler = None

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, x_scaler=None,
                 y_scaler=None):

        self._x_scaler = x_scaler
        self._y_scaler = y_scaler

        if model is None:
            model = SVR(kernel="rbf")
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # get the name of the model as string
    def to_string(self):
        pass
