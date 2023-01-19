import typing
from pathlib import Path
import numpy
import numpy.random
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import adam_v2
import visualizations

# ### Constants ########################################################################################################
IMAGE_SHAPE = IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH = (28, 28, 1)
NUM_CLASSES = 10
CLASS_NAMES = ["T_shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
EPOCHS = 75
BATCH_SIZE = 4096

# ### Data transformation ##############################################################################################
def prepare_data(train: pandas.DataFrame, test: pandas.DataFrame) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any, typing.Any, typing.Any]:
    # Convert to a valid format for Keras
    train_data = numpy.array(train, dtype=numpy.float32)
    test_data = numpy.array(test, dtype=numpy.float32)

    # Normalize pixels values and split data
    x_train = train_data[:, 1:] / 255
    y_train = train_data[:, 0]

    x_test = test_data[:, 1:] / 255
    y_test = test_data[:, 0]

    # Create a validation dataset from the training dataset
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=69420)

    return x_train, x_test, x_validate, y_train, y_test, y_validate


def training_reshape(x_train: numpy.ndarray, x_test: numpy.ndarray, x_validate: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    reshaped_x_train = x_train.reshape(x_train.shape[0], *IMAGE_SHAPE)
    reshaped_x_test = x_test.reshape(x_test.shape[0], *IMAGE_SHAPE)
    reshaped_x_validate = x_validate.reshape(x_validate.shape[0], *IMAGE_SHAPE)

    return reshaped_x_train, reshaped_x_test, reshaped_x_validate


# ### Main #############################################################################################################
def main():
    # Load the datasets
    train_data = pandas.read_csv(Path(__file__).resolve().parent / "data" / "fashion-mnist_train.csv", sep=",")
    test_data = pandas.read_csv(Path(__file__).resolve().parent / "data" / "fashion-mnist_test.csv", sep=",")

    # Prepare the dataset
    x_train, x_test, x_validate, y_train, y_test, y_validate = prepare_data(train=train_data, test=test_data)

    # Visualize the dataset
    visualizations.visualize_data(x=x_train, y=y_train, classnames=CLASS_NAMES)

    # Reshape the dataset for the training
    x_train, x_test, x_validate = training_reshape(x_train=x_train, x_test=x_test, x_validate=x_validate)

    # Define the model
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=IMAGE_SHAPE),
        MaxPooling2D(pool_size=2),  # Down sampling the output from 28x28 to 14x14
        Dropout(0.2),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(10, activation="softmax")
    ])

    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=adam_v2.Adam(learning_rate=0.001), metrics=["accuracy"])

    # Model fitting
    history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_validate, y_validate))

    # Performance plotting
    # > History rendering
    visualizations.render_history(history)

    # > Score calculation
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss : {:.4f}".format(score[0]))
    print("Test Accuracy : {:.4f}".format(score[1]))

    # > Score report per classes
    predicted_classes = numpy.argmax(model.predict(x_test), axis=-1)
    y_true = test_data.iloc[:, 0]
    target_names = ["Class {}".format(i) for i in range(NUM_CLASSES)]
    print(classification_report(y_true, predicted_classes, target_names=target_names))


if __name__ == "__main__":
    main()
