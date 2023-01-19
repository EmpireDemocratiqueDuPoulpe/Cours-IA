import typing
import numpy
from matplotlib import pyplot


# ### Data visualizations ##############################################################################################
def visualize_data(x: numpy.ndarray, y: numpy.ndarray, classnames: list[str]) -> None:
    # Create the figure
    pyplot.figure(figsize=(10, 10))

    # Add some images
    for i in range(36):
        pyplot.subplot(6, 6, (i + 1))
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.grid(False)

        pyplot.imshow(x[i].reshape((28, 28)))
        label_index = int(y[i])
        pyplot.title(classnames[label_index])

    # Show the art
    pyplot.show()


# ### Results & scores visualizations ##################################################################################
def render_history(history: typing.Any) -> None:
    # Training - Loss
    pyplot.figure(figsize=(10, 10))
    pyplot.subplot(2, 2, 1)
    pyplot.plot(history.history["loss"], label="Loss")
    pyplot.plot(history.history["val_loss"], label="Validation Loss")
    pyplot.title("Training - Loss Function")
    pyplot.legend()

    # Training - Accuracy
    pyplot.subplot(2, 2, 2)
    pyplot.plot(history.history["accuracy"], label="Accuracy")
    pyplot.plot(history.history["val_accuracy"], label="Validation Accuracy")
    pyplot.title("Train - Accuracy")
    pyplot.legend()

    # Training & Validation - Loss
    pyplot.subplot(2, 2, 3)
    epochs = range(len(history.history["accuracy"]))
    pyplot.plot(epochs, history.history["loss"], "bo", label="Training Loss")
    pyplot.plot(epochs, history.history["val_loss"], "b", label="Validation Loss")
    pyplot.title("Training and validation loss")
    pyplot.legend()

    # Training & Validation - Accuracy
    pyplot.subplot(2, 2, 4)
    pyplot.plot(epochs, history.history["accuracy"], "bo", label="Training Accuracy")
    pyplot.plot(epochs, history.history["val_accuracy"], "b", label="Validation Accuracy")
    pyplot.title("Training and Validation accuracy")
    pyplot.legend()

    pyplot.show()
