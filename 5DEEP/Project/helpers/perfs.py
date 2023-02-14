# #################################################################################################################### #
#       ./helpers/perfs.py                                                                                             #
#           Functions used to display model performance.                                                               #
# #################################################################################################################### #

# Math
import numpy

# Data
import pandas
from matplotlib import pyplot

# Model processing
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
from tensorflow import keras

# Console output
from colorama import Style


# ### Console output ###################################################################################################
def print_model_metrics(model: keras.Model, x_train: numpy.ndarray, y_train: numpy.ndarray, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
    """ Evaluates the model scores (accuracy and loss) on the train and test sets. """
    train_loss, train_acc = model.evaluate(x=x_train, y=y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)

    print(f"{Style.BRIGHT}Model metrics:{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}Training >{Style.RESET_ALL} loss={train_loss:.3f} | accuracy={(train_acc * 100):.3f}")
    print(f"{Style.BRIGHT}Testing >{Style.RESET_ALL} loss={test_loss:.3f} | accuracy={(test_acc * 100):.3f}")


def print_classification_report(model: keras.Model, x_test: numpy.ndarray, y_test: numpy.ndarray, le: LabelEncoder, classes: list[str]) -> None:
    """ Displays the classification report. """
    y_pred_one_hot = model.predict(x_test, verbose=0)  # From one-hot encoded labels ...
    y_pred_labels = numpy.argmax(y_pred_one_hot, axis=1)  # ... to label encoding
    y_test_labels = numpy.argmax(y_test, axis=1)  # Same for the true labels

    print(sklearn.metrics.classification_report(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        target_names=le.inverse_transform(numpy.arange(len(classes)))
    ))


# ### Graphs ###########################################################################################################
def plot_loss_curve(history: dict) -> None:
    """ Plot the training loss and the validation loss curves. """
    # Get the minimum loss value
    min_loss = min(history["loss"])
    min_loss_epoch = history["loss"].index(min_loss)

    min_val_loss = min(history["val_loss"])
    min_val_loss_epoch = history["val_loss"].index(min_val_loss)

    # Plot
    pyplot.figure(figsize=(12, 4))
    loss_curve, = pyplot.plot(history["loss"])  # Loss curve
    val_loss_curve, = pyplot.plot(history["val_loss"], linestyle="--")  # Validation loss curve
    min_loss_point, = pyplot.plot(min_loss_epoch, min_loss, marker="o", markersize=4)  # Plot a point at the minimum loss value
    min_val_loss_point, = pyplot.plot(min_val_loss_epoch, min_val_loss, marker="o", markersize=4)  # Plot a point at the minimum validation loss value

    pyplot.title("Loss per epochs")
    pyplot.xticks(numpy.arange(start=0, stop=len(history["loss"]), step=5), rotation=90)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend(
        handles=[loss_curve, val_loss_curve, min_loss_point, min_val_loss_point],
        labels=["Train", "Validation", f"{min_loss:.3f}", f"{min_val_loss:.3f}"]
    )
    pyplot.show()


def plot_accuracy_curve(history: dict) -> None:
    """ Plot the training accuracy and the validation accuracy curves. """
    # Get the maximum accuracy value
    max_acc = max(history["accuracy"])
    max_acc_epoch = history["accuracy"].index(max_acc)

    max_val_acc = max(history["val_accuracy"])
    max_val_acc_epoch = history["val_accuracy"].index(max_val_acc)

    # Plot
    pyplot.figure(figsize=(12, 4))
    acc_curve, = pyplot.plot(history["accuracy"])  # Accuracy curve
    val_acc_curve, = pyplot.plot(history["val_accuracy"], linestyle="--")  # Validation accuracy curve
    max_acc_point, = pyplot.plot(max_acc_epoch, max_acc, marker="o", markersize=4)  # Plot a point at the maximum accuracy value
    max_val_acc_point, = pyplot.plot(max_val_acc_epoch, max_val_acc, marker="o", markersize=4)  # Plot a point at the maximum validation accuracy value

    pyplot.title("Accuracy per epochs")
    pyplot.xticks(numpy.arange(start=0, stop=len(history["accuracy"]), step=5), rotation=90)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy")
    pyplot.legend(
        handles=[acc_curve, val_acc_curve, max_acc_point, max_val_acc_point],
        labels=["Train", "Validation", f"{max_acc:.3f}", f"{max_val_acc:.3f}"]
    )
    pyplot.show()


def plot_confusion_matrix(model: keras.Model, x_test: numpy.ndarray, y_test: numpy.ndarray, le: LabelEncoder, classes: list[str]) -> None:
    """ Plot the confusion matrix. """
    y_pred_one_hot = model.predict(x_test, verbose=0)  # From one-hot encoded labels ...
    y_pred_labels = numpy.argmax(y_pred_one_hot, axis=1)  # ... to label encoding
    y_test_labels = numpy.argmax(y_test, axis=1)  # Same for the true labels
    matrix = sklearn.metrics.confusion_matrix(y_true=y_test_labels, y_pred=y_pred_labels)

    # Plot the confusion matrix
    matrix_plot = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=le.classes_)
    matrix_plot.plot(cmap="plasma")
    pyplot.xticks(rotation=90)
    pyplot.grid(visible=False)
    pyplot.show()

    # Print the performance per class
    accuracies_per_class = []

    for index in range(0, matrix.shape[0]):
        correct_predictions = matrix[index][index].astype(int)
        total_predictions = matrix[index].sum().astype(int)
        accuracies_per_class.append(f"{round(((correct_predictions / total_predictions) * 100), 2)} %")

    print(pandas.DataFrame(data={"Class": classes, "Accuracy": accuracies_per_class}))
