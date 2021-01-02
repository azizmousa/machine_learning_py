from regression_model import RegressionModel
from sklearn.tree import DecisionTreeRegressor


class RegressionTreeModel(RegressionModel):

    def __init__(self, x_train=None, y_train=None, x_validation=None, y_validation=None, model=None, random_state=0):
        self._random_state = random_state
        if model is None:
            model = DecisionTreeRegressor(random_state=random_state)
        super().__init__(x_train, y_train, x_validation, y_validation, model)

    # create_model function resposible for initialize the Regressin model and fit it with the data
    def create_model(self):
        pass

    # get the name of the model as string
    def to_string(self):
        pass
