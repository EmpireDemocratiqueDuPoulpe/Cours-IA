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
from sklearn.linear_model import Perceptron
from sklearn import metrics
from ManualPerceptron import ManualPerceptron


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/HeartDiseaseUCI.csv")

    print(f"\n{Fore.GREEN}#### First look ########")
    print(f"Shape: {Fore.LIGHTGREEN_EX}{data.shape}")
    print(data.head())
    print(data.dtypes)

    # Replace the values of the "num" column
    print(f"\n{Fore.GREEN}#### Replace the values of the \"num\" column ########")
    data["num"] = (data["num"] > 0).astype(int)

    # Splitting dataset
    print(f"\n{Fore.GREEN}#### Splitting dataset ########")
    data_x = data.drop(["num"], axis=1)
    data_y = data["num"]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20)

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

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Get models
    models = {
        "manual_perceptron": {
            "name": "Manual Perceptron",
            "hyper_params": None
        },
        "sklearn_perceptron": {
            "name": "SKLearn Perceptron",
            "hyper_params": None
        },
    }

    for model_key in models:
        # Start of model processing
        model_start_time = timer()
        model_infos = models[model_key]

        # Get model
        print(f"\n{Fore.GREEN}#### Get model ({model_infos['name']}) ########")

        if model_infos["name"] == "Manual Perceptron":
            continue
            model, scores = None, None
        elif model_infos["name"] == "SKLearn Perceptron":
            # Model training
            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("tree", Perceptron())
            ])
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)

            # Model scores
            accuracy_score = round(metrics.accuracy_score(y_test, prediction), 2)
            model_train_score = round(model.score(x_train, y_train) * 100, 2)
            model_test_score = round(model.score(x_test, y_test) * 100, 2)

            print((
                f"Best achieved accuracy: {Fore.LIGHTGREEN_EX}{accuracy_score}"
                f"{Fore.WHITE}{Style.DIM} (train: {model_train_score}%"
                f", test: {model_test_score}%){Style.RESET_ALL}"
            ))
        else:
            raise NameError(f"Invalid model name: {model_key}")

        # > Quality measurements (Grid Search CV)
        if model_infos["hyper_params"] is not None:
            pass  # Not useful in this situation

        # End of model processing
        model_end_time = timer()
        model_elapsed_time = timedelta(seconds=model_end_time - model_start_time)
        print(f"\n{Fore.GREEN}Successful processing of {model_infos['name']} model in {model_elapsed_time}.")

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of heart disease dataset in {elapsed_time}.")


if __name__ == '__main__':
    colorama.init(autoreset=True)
    main()
