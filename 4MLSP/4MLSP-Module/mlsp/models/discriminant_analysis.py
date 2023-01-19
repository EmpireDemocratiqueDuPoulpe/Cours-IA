# #################################################################################################################### #
#       discriminant_analysis.py                                                                                       #
#           Models based on discriminant analysis.                                                                     #
# #################################################################################################################### #

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from . import common


def linear_model(preprocessor, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("linear_discriminant", LinearDiscriminantAnalysis())
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
