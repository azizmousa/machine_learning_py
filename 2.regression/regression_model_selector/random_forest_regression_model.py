from regression_model import RegressionModel
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressionModel(RegressionModel):

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, n_estimators=1,
                 random_state=0):
        self._random_state = random_state
        if model is None:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # get the name of the model as string
    def to_string(self):
        pass
