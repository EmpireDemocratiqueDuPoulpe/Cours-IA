from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/winequality.csv")

    print(f"\n{Fore.GREEN}#### First look ########")
    print(f"Shape: {Fore.LIGHTGREEN_EX}{data.shape}")
    print(data.head())
    print(data.dtypes)

    # Splitting dataset
    print(f"\n{Fore.GREEN}#### Splitting dataset ########")
    data_x = data.drop(["quality"], axis=1)
    data_y = data["quality"]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)

    # Transform numeric and categorical values
    print(f"\n{Fore.GREEN}#### Transform numeric and categorical values ########")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    categorical_features = ["type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Model
    print(f"\n{Fore.GREEN}#### Model (Multilayer Perceptron - Regression) ########")
    train_data_x = x_train[numeric_features]
    train_data_y = y_train
    test_data_x = x_test[numeric_features]
    test_data_y = y_test

    tensorflow.convert_to_tensor(train_data_x)
    tensorflow.convert_to_tensor(train_data_y)
    tensorflow.convert_to_tensor(test_data_x)
    tensorflow.convert_to_tensor(test_data_y)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=test_data_x.shape[1]))
    model.add(Dense(1))

    model.predict(test_data_x, batch_size=32)
    predictions = (model.predict(test_data_x) > 0.5).astype("int32")  # replacement for "predict_classes"

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    model.fit(train_data_x, train_data_y, batch_size=32, epochs=15, verbose=1, validation_data=(test_data_x, test_data_y))

    score = model.evaluate(test_data_x, test_data_y,  batch_size=32)
    print(f"Model score: {score}")

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of wine quality dataset in {elapsed_time}.")


if __name__ == '__main__':
    colorama.init(autoreset=True)
    main()
