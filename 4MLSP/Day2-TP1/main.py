import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Fore
import pandas
import numpy
import scipy.stats as scistats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import mlsp

# Constants
data = None


def get_args():
    usable_models = ["KNeighbors", "LinearSVC", "SVC", "LDA", "LogisticRegression", "DecisionTree"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, help=f"Which model to use: {usable_models}")
    args = parser.parse_args()

    if not args.model:
        sys.stderr.write(
            f"Missing arguments: --model <model_name>.\nUse one of the following: {', '.join(usable_models)}")
        exit(1)
    elif args.model not in usable_models:
        sys.stderr.write(f"Invalid model name \"{args.model}\".\nUse one of the following: {', '.join(usable_models)}")
        exit(1)

    return args


def process_model(model_name, preprocessor, x_train, y_train, x_test, y_test):
    model = None

    if model_name == "KNeighbors":
        # KNeighbors (KNeighborsClassifier)
        mlsp.misc.print_title("Find best model (KNeighbors)")
        model, scores = mlsp.models.neighbors.k_neighbors_model(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            range_min=1, range_max=15
        )

        # > Quality measurements (Grid Search CV)
        mlsp.misc.print_title("> Quality measurements (Grid Search CV)", char="~")
        grid_params = {
            "k_neighbors__n_neighbors": numpy.arange(1, 20),
            "k_neighbors__metric": ["euclidean", "manhattan", "minkowski"]
        }
        grid = GridSearchCV(model, grid_params, cv=10)
        model = mlsp.search.best_model(
            model, search=grid,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            train_score=scores["train_score"], test_score=scores["test_score"]
        )

        # > Quality measurements (Randomized Search CV)
        mlsp.misc.print_title("> Quality measurements (Randomized Search CV)", char="~")
        random_search_params = {
            "k_neighbors__weights": ["uniform", "distance"],
            "k_neighbors__n_neighbors": numpy.arange(1, 20),
            "k_neighbors__leaf_size": numpy.arange(1, 100),
            "k_neighbors__metric": ["euclidean", "manhattan", "minkowski"]
        }
        random_search = RandomizedSearchCV(model, random_search_params, cv=10)
        model = mlsp.search.best_model(
            model, search=random_search,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            train_score=scores["train_score"], test_score=scores["test_score"]
        )
    elif model_name == "LinearSVC":
        # LinearSVC
        mlsp.misc.print_title("Get model (LinearSVC)")
        model, scores = mlsp.models.svm.linear_svc_model(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            max_iter=2000
        )
    elif model_name == "SVC":
        # SVC
        mlsp.misc.print_title("Get model (SVC)")
        model, scores = mlsp.models.svm.svc_model(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            max_iter=-1
        )

        # > Quality measurements (Grid Search CV)
        mlsp.misc.print_title("> Quality measurements (Grid Search CV)", char="~")
        grid_params = {
            "svc__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "svc__gamma": ["scale", "auto"],
        }
        grid = GridSearchCV(model, grid_params, cv=10)
        model = mlsp.search.best_model(
            model, search=grid,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            train_score=scores["train_score"], test_score=scores["test_score"]
        )
    elif model_name == "LDA":
        # LDA (LinearDiscriminantAnalysis)
        mlsp.misc.print_title("Get model (LDA)")
        model, scores = mlsp.models.discriminant_analysis.linear_model(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )
    elif model_name == "LogisticRegression":
        # LogisticRegression
        mlsp.misc.print_title("Get model (LogisticRegression)")
        model, scores = mlsp.models.linear_model.logistic_regression_model(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )
    elif model_name == "DecisionTree":
        # DecisionTree
        mlsp.misc.print_title("Get model (DecisionTree)")
        model, scores = mlsp.models.tree.decision(
            preprocessor,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )

        global data
        # tree_export = sklearn.tree.export_graphviz(model.named_steps["tree"], out_file=None, feature_names=data.columns[1:-1])
        # pgv.AGraph(tree_export)

    return model


def main():
    start_time = timer()

    # Get args
    args = get_args()

    # Read CSV
    global data
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Get model
    model = process_model(args.model, preprocessor, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # Curves
    mlsp.misc.print_title("Curves")
    mlsp.metrics.roc_curve(model, x_test=x_test, y_test=y_test, name=args.model)
    mlsp.metrics.precision_recall_curve(model, x_test=x_test, y_test=y_test, name=args.model)

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of {args.model} model in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
