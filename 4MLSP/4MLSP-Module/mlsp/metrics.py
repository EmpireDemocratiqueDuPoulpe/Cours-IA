# #################################################################################################################### #
#       metrics.py                                                                                                     #
#           Show models metrics based on sklearn.metrics functions and matplotlib.                                     #
# #################################################################################################################### #

from matplotlib import pyplot
from sklearn import metrics


def roc_curve(model, x_test, y_test, name=""):
    prediction = model.predict(x_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
    roc_auc = metrics.auc(fpr, tpr)

    pyplot.title(f"Receiver Operating Characteristic (ROC){f' - {name}' if name else ''}")
    pyplot.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    pyplot.legend(loc="lower right")
    pyplot.plot([0, 1], [0, 1], "r--")
    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])
    pyplot.xlabel("False Positive Rate (FPR)")
    pyplot.ylabel("True Positive Rate (TPR)")
    pyplot.show()


def precision_recall_curve(model, x_test, y_test, name):
    prd = metrics.PrecisionRecallDisplay.from_estimator(model, x_test, y_test, name=name)
    prd.ax_.set_title("2-class Precision-Recall curve")
    pyplot.show()


def confusion_matrix_from_predictions(y_test, y_pred):
    cm = metrics.ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, normalize="true")
    _plot_confusion_matrix(cm)


def confusion_matrix_from_estimator(model, x_test, y_test):
    cm = metrics.ConfusionMatrixDisplay.from_estimator(estimator=model, X=x_test, y=y_test, normalize="true")
    _plot_confusion_matrix(cm)


# For some reason, this function plot a double confusion matrix.
# This is a confusion.
def _plot_confusion_matrix(matrix):
    matrix.plot()
    pyplot.show()
