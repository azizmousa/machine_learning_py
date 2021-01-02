from regression_model import RegressionModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(RegressionModel):

    _x_poly = []
    _degree_range = []

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, degree_range=[]):
        self._x_poly = []
        self._degree_range = degree_range

        if degree_range is None:
            poly = PolynomialFeatures(degree=1)
            self._x_poly.append(poly.fit_transform(x_train))
        else:
            self._degree_range = degree_range
            for degree in degree_range:
                poly = PolynomialFeatures(degree=degree)
                self._x_poly.append(poly.fit_transform(x_train))

        if model is None:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # get the name of the model as string
    def to_string(self):
        return "Polynomial Regreassion Model"
