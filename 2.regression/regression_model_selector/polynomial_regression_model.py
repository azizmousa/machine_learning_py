from regression_model import RegressionModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class LinearRegressionModel(RegressionModel):

    _degree_range = []
    _degreed_model = {}

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, degree_range=None):
        if degree_range is None:
            degree_range = []
        self._degree_range = degree_range
        self._degreed_model = {}
        if model is None:
            model = LinearRegression()
        super().__init__(x_train, y_train, x_validation, y_validation, model)

        # create_model function resposible for initialize the Regressin model and fit it with the data

    def create_model(self):
        if self._degree_range is []:
            print(f"Training Polynomial Regression Model of Degree [1] ....")
            x_poly_train = PolynomialFeatures(degree=1)
            self._model.fit(x_poly_train, self._y_train)
        else:
            for degree in self._degree_range:
                print(f"Training Polynomial Regression Model of Degree [{degree}] ....")
                x_poly_train = PolynomialFeatures(degree=degree)
                model = LinearRegression()
                model.fit(x_poly_train, self._y_train)
                self._degreed_model.append({degree: model})
                print(f"Training Polynomial Regression Model of Degree [{degree}] is Finished >>")

    # get the name of the model as string
    def to_string(self):
        return "Polynomial Regreassion Model"
