from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Fore
import pandas
import numpy
import scipy.stats as scistats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/HeartDiseaseUCI.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data)

    # Replace num column
    mlsp.misc.print_title("Replace \"num\" column")
    data = data.assign(num=lambda n: (n["num"] > 0).astype(int))
    data = data.rename({"num": "disease"}, axis="columns")
    print(data.head())

    # Study
    mlsp.misc.print_title("Study")
    conf_interval = scistats.t.interval(alpha=0.95, df=(len(data) - 1), loc=numpy.mean(data), scale=scistats.sem(data))
    print(f"Confidence interval: {Fore.LIGHTGREEN_EX}{conf_interval}")

    # Splitting dataset
    mlsp.misc.print_title("Splitting dataset")
    x_train, x_test, y_train, y_test = mlsp.df.split_train_test(data, y_label="disease", test_size=0.20)
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}")

    # Transform numeric and categorical values
    mlsp.misc.print_title("Transform numeric and categorical values")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))  # Not sure about "unknown"
    ])

    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    preprocessor_quantitative = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features)
        ]
    )

    preprocessor_qualitative = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    preprocessor_mixed = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Get model - GaussianNB
    mlsp.misc.print_title("Get model (GaussianNB)")
    model_quantitative, scores_quantitative = mlsp.models.naive_bayes.gaussian_model(
        preprocessor_quantitative,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    # Get model - CategoricalNB
    mlsp.misc.print_title("Get model (CategoricalNB)")
    model_qualitative, scores_qualitative = mlsp.models.naive_bayes.categorical_model(
        preprocessor_qualitative,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    # Get model - MixedNB
    # don't work :(
    # mlsp.misc.print_title("Get model (MixedNB)")
    # model_mixed, scores_mixed = mlsp.models.naive_bayes.mixed_model(
    #     preprocessor_mixed,
    #     categorical_feat=[2, 3, 8, 9, 11, 13, 15],
    #     max_categories=[2, 4, 2, 3, 2, 3, 3],
    #     x_train=x_train, y_train=y_train,
    #     x_test=x_test, y_test=y_test
    # )

    # Curves - GaussianNB
    mlsp.misc.print_title("Curves (GaussianNB)")
    mlsp.metrics.roc_curve(model_quantitative, x_test=x_test, y_test=y_test, name="GaussianNB")
    mlsp.metrics.precision_recall_curve(model_quantitative, x_test=x_test, y_test=y_test, name="GaussianNB")

    # Curves - CategoricalNB
    mlsp.misc.print_title("Curves (CategoricalNB)")
    mlsp.metrics.roc_curve(model_qualitative, x_test=x_test, y_test=y_test, name="CategoricalNB")
    mlsp.metrics.precision_recall_curve(model_qualitative, x_test=x_test, y_test=y_test, name="CategoricalNB")

    # Curves - MixedNB
    # mlsp.misc.print_title("Curves (MixedNB)")
    # mlsp.metrics.roc_curve(model_mixed, x_test=x_test, y_test=y_test, name="MixedNB")
    # mlsp.metrics.precision_recall_curve(model_mixed, x_test=x_test, y_test=y_test, name="MixedNB")

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of naive bayes classifier in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
