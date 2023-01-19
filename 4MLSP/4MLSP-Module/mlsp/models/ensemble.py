# #################################################################################################################### #
#       ensemble.py                                                                                                    #
#           Ensemble-based models (no way).                                                                            #
# #################################################################################################################### #

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from . import common


def random_forest_regressor(preprocessor, x_train, y_train, x_test, y_test, n_estimators=100, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("random_forest_regressor", RandomForestRegressor(n_estimators=n_estimators))
    ])

    return common.process_regression_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
