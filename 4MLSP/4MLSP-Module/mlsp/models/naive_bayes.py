# #################################################################################################################### #
#       naive_bayes.py                                                                                                 #
#           Models based on naive bayes analysis.                                                                      #
# #################################################################################################################### #

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import Pipeline
from mixed_naive_bayes import MixedNB
from . import common


def gaussian_model(preprocessor, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("gaussian_nb", GaussianNB())
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)


def categorical_model(preprocessor, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("categorical_nb", CategoricalNB())
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)


def mixed_model(preprocessor, categorical_feat, max_categories, x_train, y_train, x_test, y_test, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("mixed_nb", MixedNB(categorical_features=categorical_feat, max_categories=max_categories))
    ])

    return common.process_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
