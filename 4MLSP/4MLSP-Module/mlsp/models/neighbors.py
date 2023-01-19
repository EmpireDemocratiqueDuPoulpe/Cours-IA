# #################################################################################################################### #
#       neighbors.py                                                                                                   #
#           Models based on nearest neighbors analysis.                                                                #
# #################################################################################################################### #

from colorama import Fore, Style
from matplotlib import pyplot
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from . import common


def k_neighbors_model(preprocessor, x_train, y_train, x_test, y_test, range_min=1, range_max=15, verbose=False):
    models_list = []
    scores_list = []

    # Test each `k` between `range_min` and `range_max`
    print(f"{Fore.YELLOW}Testing model with `k` in range [{range_min}, {range_max}]...")
    k_range = range(range_min, range_max)

    for k in k_range:
        classifier = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("k_neighbors", KNeighborsClassifier(n_neighbors=k))
        ])
        classifier.fit(x_train, y_train)
        models_list.append(classifier)

        prediction = classifier.predict(x_test)
        score = metrics.accuracy_score(y_test, prediction)
        scores_list.append(score)

        if verbose:
            common.print_confusion_matrix(y_test, prediction)
            common.print_classification_rprt(y_test, prediction)

        print(f"{Fore.WHITE}{Style.DIM}(k={k}): {round(score, 2)}")

    # Plot the result
    print(f"{Fore.YELLOW}Finished prediction. Plotting scores...")
    pyplot.plot(k_range, scores_list)
    pyplot.xlabel("Value of K")
    pyplot.ylabel("Accuracy")
    pyplot.title("Model accuracy for k")
    pyplot.show()

    # Return the best model
    best_score = max(scores_list)
    model = models_list[scores_list.index(best_score)]
    acc_score, train_score, test_score = common.print_score(
        model,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    return model, {"accuracy_score": acc_score, "train_score": train_score, "test_score": test_score}


def k_neighbors_regressor(preprocessor, x_train, y_train, x_test, y_test, n_neighbors=5, verbose=False):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("k_neighbors_regressor", KNeighborsRegressor(n_neighbors=n_neighbors))
    ])

    return common.process_regression_model(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, verbose=verbose)
