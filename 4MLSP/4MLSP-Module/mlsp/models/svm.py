# #################################################################################################################### #
#       svm.py                                                                                                         #
#           Models based on support vector machines algorithms.                                                        #
# #################################################################################################################### #

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from . import common


def linear_svc_model(preprocessor, x_train, y_train, x_test, y_test, max_iter=1000, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("linear_svc", LinearSVC(max_iter=max_iter))
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)


def svc_model(preprocessor, x_train, y_train, x_test, y_test, max_iter=1000, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svc", SVC(max_iter=max_iter))
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
