from pathlib import Path
import time
import pandas
import colorama
from colorama import Fore, Style
import numpy
from matplotlib import pyplot
from tensorflow import keras

# ### Constants ########################################################################################################
ALL_COLUMNS = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)",
               "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)"]
FEATURE_NAMES = ["p (mbar)", "T (degC)", "VPmax (mbar)", "VPdef (mbar)", "sh (g/kg)", "rho (g/m**3)", "wv (m/s)"]
TARGET_NAME = "T (degC)"

SPLIT_FRACTION = 0.715
STEP = 6

PAST = 720
FUTURE = 72
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 10


# ### Exploratory Analysis #############################################################################################
def first_look(df: pandas.DataFrame) -> None:
    print(f"Shape: {Fore.LIGHTGREEN_EX}{df.shape}")
    print(df.head())
    print(df.dtypes)


def missing_values(df: pandas.DataFrame, keep_zeros: bool = True) -> None:
    data_count = df.shape[0] * df.shape[1]
    missing = missing_df = df.isna().sum()

    if not keep_zeros:
        missing_df = missing_df[missing_df > 0]

    missing_df = missing_df.sort_values(ascending=False).apply(lambda m: f"{m} ({round((m * 100) / df.shape[0], 2)}%)")

    print((
        f"Missing values: {Fore.LIGHTGREEN_EX}{round((missing.sum() / data_count) * 100, 2)}%\n"
        f"{Fore.WHITE}{Style.DIM}{missing_df}"
    ))


def plot_meteorology(df: pandas.DataFrame) -> None:
    plots = [
        { "key": "p (mbar)", "title": "Pressure (mbar)", "color": "blue" },
        { "key": "T (degC)", "title": "Temperature (degC)", "color": "orange" },
        { "key": "Tpot (K)", "title": "Temperature (K)", "color": "green" },
        { "key": "Tdew (degC)", "title": "Temperature relative to humidity (degC)", "color": "red" },
        { "key": "rh (%)", "title": "Relative humidity (%)", "color": "purple" },
        { "key": "VPmax (mbar)", "title": "Saturation vapor pressure (mbar)", "color": "brown" },
        { "key": "VPact (mbar)", "title": "Vapor pressure (mbar)", "color": "pink" },
        { "key": "VPdef (mbar)", "title": "Vapor pressure deficit (mbar)", "color": "gray" },
        { "key": "sh (g/kg)", "title": "Specific humidity (g/kg)", "color": "olive" },
        { "key": "H2OC (mmol/mol)", "title": "Water vapor concentration (mmol/mol)", "color": "cyan" },
        { "key": "rho (g/m**3)", "title": "Airtight (g/m**3)", "color": "blue" },
        { "key": "wv (m/s)", "title": "Wind speed (m/s)", "color": "orange" },
        { "key": "max. wv (m/s)", "title": "Maximum wind speed (m/s)", "color": "green" },
    ]

    fig, axs = pyplot.subplots(nrows=7, ncols=2, figsize=(15, 20))

    for idx in range(len(plots)):
        plot = plots[idx]
        plot_data = df[plot["key"]]

        ax = plot_data.plot(ax=axs[idx // 2, idx % 2], color=plot["color"], title=plot["title"])
        ax.legend(plot["key"])

    pyplot.suptitle("Data overview")
    pyplot.tight_layout()
    pyplot.show()


def plot_correlation_matrix(df: pandas.DataFrame) -> None:
    # Drop columns with NaN and exclude the one with less than 1 unique value.
    df = df.dropna()
    df = df[[column for column in df if df[column].nunique() > 1]]
    if df.shape[1] < 2:
        print(f"No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2")
        return

    # Correlation matrix
    corr = df.corr(numeric_only=True)

    pyplot.figure(num=None, figsize=(8, 8), dpi=80, facecolor="w", edgecolor="k")
    corr_matrix = pyplot.matshow(corr, fignum=1)
    pyplot.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    pyplot.yticks(range(len(corr.columns)), corr.columns)
    pyplot.gca().xaxis.tick_bottom()
    pyplot.colorbar(corr_matrix)
    pyplot.title("Correlation Matrix", fontsize=15)
    pyplot.show()


def plot_scatter_matrix(df: pandas.DataFrame) -> None:
    # Keep only numerical columns
    df = df.select_dtypes(include =[numpy.number])

    # Remove rows and columns that would lead to df being singular
    df = df.dropna()
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    # Get the column names used for the plot
    column_names = list(df)
    if len(column_names) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        column_names = column_names[:10]
    df = df[column_names]

    # Plot the scatter matrix
    ax = pandas.plotting.scatter_matrix(df, alpha=0.75, figsize=[20, 20], diagonal="kde")
    corrs = df.corr(numeric_only=True).values

    for i, j in zip(*pyplot.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate("Corr. coef = %.3f" % corrs[i, j], (0.8, 0.2), xycoords="axes fraction", ha="center", va="center", size=10)

    pyplot.suptitle("Scatter and Density Plot")
    pyplot.show()


# ### Preprocessing ####################################################################################################
def normalize(data: pandas.DataFrame, train_split: int) -> pandas.DataFrame:
    values = data.values
    values_mean = values[:train_split].mean(axis=0)
    values_std = values[:train_split].std(axis=0)

    return pandas.DataFrame((values - values_mean) / values_std)


# ### Post training analysis ###########################################################################################
def visualize_loss(history) -> None:
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))

    # Plot
    pyplot.figure()
    pyplot.plot(epochs, loss, "b", label="Training loss")
    pyplot.plot(epochs, val_loss, "r", label="Validation loss")
    pyplot.title("Training and Validation Loss")
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.show()


# ### Predicting #######################################################################################################
def show_prediction_plot(plot_data, delta: int, title: str) -> None:
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]

    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    future = delta if delta else 0

    # Plot
    pyplot.title(title)

    for i, val in enumerate(plot_data):
        if i:
            pyplot.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            pyplot.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    pyplot.legend()
    pyplot.xlim([time_steps[0], (future + 5) * 2])
    pyplot.xlabel("Time-Step")
    pyplot.show()


# ### Main #############################################################################################################
def main() -> None:
    # Load the dataset
    data = pandas.read_csv(Path(__file__).resolve().parent / "data" / "jena_climate_2009_2016.csv")

    # First look
    print(f"{Fore.GREEN}{Style.BRIGHT}#### First look ################################################################")
    first_look(data)


    # Missing values
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Missing values ############################################################")
    missing_values(data, keep_zeros=True)


    # Data analysis
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Data analysis #############################################################")
    plot_meteorology(data)


    # Removing incorrect data
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Removing incorrect data ###################################################")
    # print(data["wv (m/s)"][(data["wv (m/s)"] < 0)])
    # print(data["max. wv (m/s)"][(data["max. wv (m/s)"] < 0)])
    prev_shape = data.shape
    data.drop(data[data["wv (m/s)"] < 0].index, inplace=True)
    data.drop(data[data["max. wv (m/s)"] < 0].index, inplace=True)
    new_shape = data.shape
    print(f"Removed {prev_shape[0] - new_shape[0]} rows.")


    # Data analysis II - A New Hope
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Data analysis II - A New Hope #############################################")
    plot_meteorology(data)
    plot_correlation_matrix(data)
    # plot_scatter_matrix(data) very very very very very long.


    # Normalization
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Normalization #############################################################")
    features = data[FEATURE_NAMES]
    features.index = data["Date Time"]
    print(features.head())

    train_split = int(SPLIT_FRACTION * int(data.shape[0]))
    features = normalize(features, train_split=train_split)
    print(features.head())


    # Split into training/testing datasets
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Split into training/testing datasets ######################################")
    train_data = features.loc[0: train_split - 1]
    val_data = features.loc[train_split:]

    # Training dataset
    start = PAST + FUTURE
    end = start + train_split
    sequence_length = int(PAST / STEP)

    x_train = train_data[[i for i in range(7)]].values
    y_train = features.iloc[start:end][[1]]

    train_dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=x_train, targets=y_train, sequence_length=sequence_length, sampling_rate=STEP, batch_size=BATCH_SIZE
    )

    # Validation dataset
    x_end = len(val_data) - PAST - FUTURE
    label_start = train_split + PAST + FUTURE

    x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
    y_val = features.iloc[label_start:][[1]]

    val_dataset = keras.preprocessing.timeseries_dataset_from_array(
        data=x_val, targets=y_val, sequence_length=sequence_length, sampling_rate=STEP, batch_size=BATCH_SIZE,
    )

    # Console output
    inputs, targets = None, None
    for batch in train_dataset.take(1):
        inputs, targets = batch

    if (inputs is None) or (targets is None):
        print("Well, that's unfortunate. It crashed.")
        return

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)


    # Training
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Training ##################################################################")
    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    # Model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    model.summary()

    # Checkpoint saving
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / f"model_checkpoint_{time.time_ns() // 1_000_000}.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss", filepath=checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True
    )

    # Fitting
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[es_callback, modelckpt_callback])


    # Post training analysis
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Post training analysis ####################################################")
    visualize_loss(history)


    # Predicting
    print(f"{Fore.GREEN}{Style.BRIGHT}#### Predicting ################################################################")
    for x, y in val_dataset.take(5):
        show_prediction_plot(
            plot_data=[x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            delta=12,
            title="Single Step Prediction",
        )


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
