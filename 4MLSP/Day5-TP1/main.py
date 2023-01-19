from timeit import default_timer as timer
from datetime import timedelta
import colorama
import numpy
from colorama import Style, Fore
import pandas
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/diamonds.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data)

    # Merge [x, y, z] into `volume`
    mlsp.misc.print_title("Merge [x, y, z] into `volume`")
    data["volume"] = data["x"] * data["y"] * data["z"]
    data.drop(columns=["x", "y", "z"], inplace=True)
    print(data.sample(n=5))

    # Study
    mlsp.misc.print_title("Study")
    features = ["carat", "depth", "table", "volume", "price"]

    analyze = data.describe().T
    analyze["dispersion"] = data[features].std() / data[features].mean() * 100
    analyze["standard error"] = data[features].sem()
    analyze["skewness"] = data[features].skew()
    analyze["kurtosis"] = data[features].kurtosis()

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    is_outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))
    analyze["outliers"] = is_outliers.sum(axis=0) / data.shape[0] * 100  # Percent of outliers per column

    pandas.set_option("display.max_columns", None)
    print(f"Analyze:\n{Style.DIM}{Fore.WHITE}{analyze}")
    pandas.set_option("display.max_columns", 5)

    # Splitting dataset
    mlsp.misc.print_title("Splitting dataset")
    x_train, x_test, y_train, y_test = mlsp.df.split_train_test(data, y_label="price", test_size=0.20)
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
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["carat", "depth", "table", "volume"]
    categorical_features = ["cut", "color", "clarity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Get models
    models = {
        "linear_regression": {
            "name": "Linear Regression",
            "hyper_params": None
        },
        "k_neighbors_regression": {
            "name": "KNeighbors Regression",
            "hyper_params": {
                "k_neighbors_regressor__n_neighbors": numpy.arange(1, 20),
                "k_neighbors_regressor__metric": ["euclidean", "manhattan", "minkowski"]
            }
        },
        "decision_tree_regressor": {
            "name": "Decision Tree Regressor",
            "hyper_params": {
                "tree_regressor__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "tree_regressor__min_samples_split": numpy.arange(2, 3),
            }
        },
        "random_forest_regressor": {
            "name": "Random Forest Regressor",
            "hyper_params": {
                "random_forest_regressor__n_estimators": numpy.arange(100, 105),
                "random_forest_regressor__criterion": ["squared_error", "absolute_error", "poisson"],
            }
        }
    }

    for model_key in models:
        # Start of model processing
        model_start_time = timer()
        model_infos = models[model_key]

        # Get model
        mlsp.misc.print_title(f"Get model ({model_infos['name']})")

        if model_infos["name"] == "Linear Regression":
            model, scores = mlsp.models.linear_model.linear_regression_model(
                preprocessor,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test
            )
        elif model_infos["name"] == "KNeighbors Regression":
            model, scores = mlsp.models.neighbors.k_neighbors_regressor(
                preprocessor,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test
            )
        elif model_infos["name"] == "Decision Tree Regressor":
            model, scores = mlsp.models.tree.regressor(
                preprocessor,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test
            )
        elif model_infos["name"] == "Random Forest Regressor":
            model, scores = mlsp.models.ensemble.random_forest_regressor(
                preprocessor,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test
            )
        else:
            raise NameError(f"Invalid model name: {model_key}")

        # > Quality measurements (Grid Search CV)
        if model_infos["hyper_params"] is not None:
            mlsp.misc.print_title("> Quality measurements (Grid Search CV)", char="~")
            grid = GridSearchCV(model, model_infos["hyper_params"], cv=10, verbose=1)
            mlsp.search.best_model(
                model, search=grid,
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                train_score=scores["train_score"], test_score=scores["test_score"]
            )

        # End of model processing
        model_end_time = timer()
        model_elapsed_time = timedelta(seconds=model_end_time - model_start_time)
        print(f"\n{Fore.GREEN}Successful processing of {model_infos['name']} model in {model_elapsed_time}.")

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of diamonds dataset in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
