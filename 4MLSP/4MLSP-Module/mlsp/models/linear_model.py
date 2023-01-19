# #################################################################################################################### #
#       lm.py                                                                                                          #
#           Linear models. WATCH OUT A LINE --> |                                                                      #
# #################################################################################################################### #

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from . import common


def logistic_regression_model(preprocessor, x_train, y_train, x_test, y_test, max_iter=100, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("logistic_regression", LogisticRegression(max_iter=max_iter))
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)


def linear_regression_model(preprocessor, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("linear_regression", LinearRegression())
    ])

    return common.process_regression_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
