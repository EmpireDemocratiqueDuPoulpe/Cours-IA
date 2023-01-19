# #################################################################################################################### #
#       tree.py                                                                                                        #
#           Models based on tree classification and regression.                                                        #
# #################################################################################################################### #

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from . import common


def decision(preprocessor, x_train, y_train, x_test, y_test, criterion="gini", verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(criterion=criterion))
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)


def regressor(preprocessor, x_train, y_train, x_test, y_test, criterion="squared_error", verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("tree_regressor", DecisionTreeRegressor(criterion=criterion))
    ])

    return common.process_regression_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
