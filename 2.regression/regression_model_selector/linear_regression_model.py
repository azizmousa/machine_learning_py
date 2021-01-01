from regression_model import RegressionModel
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(RegressionModel):

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None):
        if model is None:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # get the name of the model as string
    def to_string(self):
        return "Linear Regreassion Model"

