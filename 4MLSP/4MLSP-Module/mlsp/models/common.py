# #################################################################################################################### #
#       common.py                                                                                                      #
#           Common functions used by other functions in this folder.                                                   #
# #################################################################################################################### #

import warnings
from colorama import Fore, Style
import numpy
from sklearn import metrics
import scipy.stats


def process_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    if verbose:
        print_confusion_matrix(y_test, prediction)
        print_classification_rprt(y_test, prediction)

    # Return the best model
    acc_score, train_score, test_score = print_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy_score": acc_score, "train_score": train_score, "test_score": test_score}


def process_regression_model(model, x_train, y_train, x_test, y_test, verbose=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    # Return the best model
    train_score, test_score = print_regression_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        prediction=prediction
    )
    return model, {"accuracy_score": None, "train_score": train_score, "test_score": test_score}


def print_confusion_matrix(y_test, prediction):
    print(f"Confusion matrix: {Fore.LIGHTGREEN_EX}{metrics.confusion_matrix(y_test, prediction)}")


def print_classification_rprt(y_test, prediction):
    print(f"Classification report: {Fore.LIGHTGREEN_EX}{metrics.classification_report(y_test, prediction)}")


def print_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    accuracy_score = round(metrics.accuracy_score(y_test, prediction), 2)
    model_train_score = round(model.score(x_train, y_train) * 100, 2)
    model_test_score = round(model.score(x_test, y_test) * 100, 2)

    print((
        f"Best achieved accuracy: {Fore.LIGHTGREEN_EX}{accuracy_score}"
        f"{Fore.WHITE}{Style.DIM} (train: {model_train_score}%"
        f", test: {model_test_score}%){Style.RESET_ALL}"
    ))

    return accuracy_score, model_train_score, model_test_score


def print_regression_score(model, x_train, y_train, x_test, y_test, prediction=None):
    if prediction is None:
        prediction = model.predict(x_test)

    model_train_score = round(model.score(x_train, y_train) * 100, 2)
    model_test_score = round(model.score(x_test, y_test) * 100, 2)

    mae = round(metrics.mean_absolute_error(y_true=y_test, y_pred=prediction), 3)
    rmse = round(numpy.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=prediction)), 3)
    mape = round(metrics.mean_absolute_percentage_error(y_true=y_test, y_pred=prediction) * 100, 3)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:
            statistic, pvalue = scipy.stats.shapiro(prediction - y_test)
        except UserWarning as warn:
            print(f"{Fore.YELLOW}Warning: {warn}")
            warnings.filterwarnings("ignore")
            statistic, pvalue = scipy.stats.shapiro(prediction - y_test)

    print((
        f"Best achieved accuracy:"
        f" (train: {Fore.LIGHTGREEN_EX}{model_train_score}%{Fore.RESET}"
        f", test: {Fore.LIGHTGREEN_EX}{model_test_score}%{Fore.RESET})\n"
        
        f"Regression model scores:"
        f" (MAE: {Fore.LIGHTGREEN_EX}{mae}{Fore.RESET}"
        f", RMSE: {Fore.LIGHTGREEN_EX}{rmse}{Fore.RESET}"
        f", MAPE: {Fore.LIGHTGREEN_EX}{mape}%{Fore.RESET})\n"

        f"Normality test:"
        f" (pvalue: {Fore.LIGHTGREEN_EX}{round(pvalue, 1)}{Fore.RESET}"
        f", statistic: {Fore.LIGHTGREEN_EX}{round(statistic, 3)}{Fore.RESET})"
    ))

    return model_train_score, model_test_score
