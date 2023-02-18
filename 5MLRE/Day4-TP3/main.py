from timeit import default_timer as timer
from datetime import timedelta
import pandas
import colorama
from colorama import Fore, Style
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# ### Constants ########################################################################################################
RANDOM_STATE = 2077


# ### Functions - Data exploration #####################################################################################
def missing_values(df: pandas.DataFrame, keep_zeros: bool = True) -> None:
    data_count = df.shape[0] * df.shape[1]
    missing = missing_df = df.isna().sum()

    if not keep_zeros:
        missing_df = missing_df[missing_df > 0]

    missing_df = missing_df.sort_values(ascending=False).apply(lambda m: f"{m} ({round((m * 100) / df.shape[0], 2)}%)")

    print((
        f"{Style.BRIGHT}Missing values:{Style.RESET_ALL} {round((missing.sum() / data_count) * 100, 2)}%\n"
        f"{Style.DIM}{Fore.WHITE}{missing_df}{Style.RESET_ALL}"
    ))


def duplicated_values(df: pandas.DataFrame) -> None:
    data_count = df.shape[0] * df.shape[1]
    duplicated = df.duplicated().sum()

    print(f"{Style.BRIGHT}Duplicated values:{Style.RESET_ALL} {duplicated} ({round((duplicated.sum() / data_count) * 100, 2)}%)")


# ### Main #############################################################################################################
def main() -> None:
    data = pandas.read_csv("./data/churn.csv")

    # Q1 - Quick data exploration
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q1 - Quick data exploration ##############################################{Style.RESET_ALL}")
    print(f"Shape: {data.shape}")
    print(data.head(n=5))
    print(data.dtypes)

    missing_values(df=data, keep_zeros=True)
    duplicated_values(df=data)

    # Q2 - Data preprocessing
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q2 - Data preprocessing ##################################################{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}Filtering columns...{Style.RESET_ALL}")
    data = data[["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]]

    print(f"{Style.BRIGHT}Converting the columns type...{Style.RESET_ALL}")
    data[["Geography", "Gender"]] = data[["Geography", "Gender"]].astype("category")
    data[["HasCrCard", "IsActiveMember", "Exited"]] = data[["HasCrCard", "IsActiveMember", "Exited"]].astype(bool)
    print(data.dtypes)

    # Q3 & Q4 - Models training
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q3 & Q4 - Models training ################################################{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}Splitting the dataset...{Style.RESET_ALL}")
    data_x = data.drop(["Exited"], axis=1)
    data_y = data["Exited"]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20)
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}{Fore.RESET}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}{Fore.RESET}")

    print(f"{Style.BRIGHT}Setting up the preprocessor...{Style.RESET_ALL}")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    categorical_features = ["Geography", "Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder="passthrough"
    )

    print(f"{Style.BRIGHT}Testing multiple models...{Style.RESET_ALL}")
    models = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "instance": LogisticRegression(max_iter=100),
            "hyper_params": {
                "logistic_regression__penalty": ["l2", "none"]
            }
        },
        "linear_regression": {
            "name": "Linear Regression",
            "instance": LinearRegression(),
            "hyper_params": None
        },
        "k_neighbors_regression": {
            "name": "KNeighbors Regression",
            "instance": KNeighborsRegressor(),
            "hyper_params": {
                "k_neighbors_regression__n_neighbors": numpy.arange(1, 20),
                "k_neighbors_regression__metric": ["euclidean", "manhattan", "minkowski"]
            }
        },
        "decision_tree_regressor": {
            "name": "Decision Tree Regressor",
            "instance": DecisionTreeRegressor(),
            "hyper_params": {
                "decision_tree_regressor__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "decision_tree_regressor__min_samples_split": numpy.arange(2, 3),
            }
        },
        "random_forest_regressor": {
            "name": "Random Forest Regressor",
            "instance": RandomForestRegressor(),
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

        print(f"\n\nTesting \"{model_infos['name']}\"...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            (model_key, model_infos["instance"])
        ])

        # > Quality measurements (Grid Search CV)
        if model_infos["hyper_params"] is not None:
            print(f"{Style.DIM}{Fore.WHITE}> Quality measurements (Grid Search CV){Style.RESET_ALL}")
            grid = GridSearchCV(pipeline, model_infos["hyper_params"], cv=10, verbose=1)

            grid.fit(x_train, y_train)
            print(f"{Style.DIM}{Fore.WHITE}Best params for model:{Fore.RESET} {Fore.LIGHTGREEN_EX}{grid.best_params_}{Style.RESET_ALL}")
            model = grid.best_estimator_

            train_score = round(model.score(x_train, y_train) * 100, 2)
            test_score = round(model.score(x_test, y_test) * 100, 2)
        else:
            pipeline.fit(x_train, y_train)
            train_score = round(pipeline.score(x_train, y_train) * 100, 2)
            test_score = round(pipeline.score(x_test, y_test) * 100, 2)

        # End of model processing
        model_end_time = timer()
        model_elapsed_time = timedelta(seconds=model_end_time - model_start_time)
        print(f"Successful processing of {model_infos['name']} model in {model_elapsed_time}.")
        print((
            f"Model score:\n"
            f"\t{Fore.WHITE}{Style.DIM}∟ train → {train_score}%{Style.RESET_ALL}\n"
            f"\t{Fore.WHITE}{Style.DIM}∟ test → {test_score}%{Style.RESET_ALL}"
        ))


if __name__ == "__main__":
    # Packages init
    colorama.init(autoreset=False)
    numpy.random.seed(RANDOM_STATE)

    # Main
    main()