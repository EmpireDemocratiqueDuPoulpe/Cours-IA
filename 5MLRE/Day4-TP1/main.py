import pandas
import colorama
from colorama import Fore, Style
import surprise
from surprise import accuracy as surp_acc
import numpy
from collections import defaultdict

# ### Constants ########################################################################################################
RANDOM_STATE = 2077


# ### Metrics ##########################################################################################################
def map_predictions(predictions: list) -> defaultdict[str, list]:
    """ Map the predictions to each user """
    user_to_predictions = defaultdict(list)

    for user_id, item_id, true_rating, estimated_rating, _ in predictions:
        user_to_predictions[user_id].append((item_id, estimated_rating, true_rating))

    return user_to_predictions


def get_top_n(predictions: list, n: int = 5, min_rating: float = 0.0) -> dict[str, list]:
    """ Return the top-N recommendation for each user """
    # Map the predictions to each user
    top_n = map_predictions(predictions)

    # Compute the top N of each user
    for user_id, user_ratings in top_n.items():
        # Sort the user ratings by the estimated value
        user_ratings.sort(key=(lambda x: x[1]), reverse=True)

        # Cut down the top
        top_n[user_id] = user_ratings[:n]

        # Filter the ratings below the threshold
        top_n[user_id] = list(filter(lambda x: x[1] >= min_rating, user_ratings))

    return dict(top_n)


def get_hit_rate(top_n: dict[str, list], left_out_predictions: list) -> float:
    """ Compute the overall hit rate """
    hits = 0
    total = 0

    for user_id, item_id, true_rating, estimated_rating, _ in left_out_predictions:
        hit = False  # Is the movie in the predicted top N for this user?

        for top_item_id, _, _ in top_n[user_id]:
            if top_item_id == item_id:
                hit = True
                break

        # Add the rating
        total += 1
        if hit:
            hits += 1

    return hits / total


def get_cumulative_hit_rate(top_n: dict[str, list], left_out_predictions: list, min_rating: float = 4.0) -> float:
    """ Compute the overall cumulative hit rate """
    hits = 0
    total = 0

    for user_id, item_id, true_rating, estimated_rating, _ in left_out_predictions:
        if true_rating >= min_rating:  # Filter out items depending on the true rating
            hit = False  # Is the movie in the predicted top N for this user?

            for top_item_id, _, _ in top_n[user_id]:
                if top_item_id == item_id:
                    hit = True
                    break

            # Add the rating
            total += 1
            if hit:
                hits += 1

    return hits / total


def get_average_reciprocal_hit_rate(top_n: dict[str, list], left_out_predictions: list) -> float:
    """ Compute the average reciprocal hit rate """
    summary = 0
    total = 0

    for user_id, item_id, true_rating, estimated_rating, _ in left_out_predictions:
        # Get the hit rank of the movie. The lower, the better
        hit_rank = 0
        rank = 0

        for top_item_id, _, _ in top_n[user_id]:
            rank += 1

            if top_item_id == item_id:
                hit_rank = rank
                break

        # Add the hit rank
        total += 1

        if hit_rank > 0:
            summary += 1.0 / hit_rank

    return summary / total


def get_user_coverage(top_n: dict[str, list], users_count: int, min_rating: float = 4.0) -> float:
    """ Compute the user coverage """
    hits = 0

    for user_id in top_n.keys():
        hit = False

        for top_item_id, top_estimated_rating, top_true_rating in top_n[user_id]:
            if top_estimated_rating >= min_rating:
                hit = True
                break

        if hit:
            hits += 1

    return hits / users_count


# ### Main #############################################################################################################
def main() -> None:
    # Q1 - Load datasets
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q1 - Load datasets #######################################################{Style.RESET_ALL}")
    data = surprise.Dataset.load_builtin(name="ml-100k", prompt=True)

    # Q2 - Slope One algorithm
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q2 - Slope One algorithm #################################################{Style.RESET_ALL}")
    print("Model initialization...")
    model_slope_one = surprise.SlopeOne()
    data_iterator = surprise.model_selection.KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)  # Initialize the data iterator
    i = 0

    for data_train, data_test in data_iterator.split(data):
        print(f"\nProcessing fold {i + 1}")
        model_slope_one.fit(data_train)  # Train the model
        predictions_slope_one = model_slope_one.test(data_test)  # Predict using the test dataset

        # Accuracy calculation
        surp_acc.rmse(predictions_slope_one)
        surp_acc.mae(predictions_slope_one)

        i += 1

    # Metrics
    print(f"Calculating model metrics...{Fore.WHITE}{Style.DIM}")
    data_train, data_test = surprise.model_selection.train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)  # Split the dataset
    predictions = model_slope_one.test(data_test)
    print(f"{Style.RESET_ALL}")

    top_n = get_top_n(predictions=predictions, n=10, min_rating=2.5)
    print(f"Built top N for each user (n=10, min_rating=2.5)")
    print(f"{Style.BRIGHT}Hit rate:{Style.NORMAL} {get_hit_rate(top_n=top_n, left_out_predictions=predictions) * 100}%")
    print(f"{Style.BRIGHT}Cumulative hit rate (min_rating=2.5):{Style.NORMAL} {get_cumulative_hit_rate(top_n=top_n, left_out_predictions=predictions, min_rating=2.5) * 100}%")
    print(f"{Style.BRIGHT}Average reciprocal hit rate:{Style.NORMAL} {get_average_reciprocal_hit_rate(top_n=top_n, left_out_predictions=predictions)}")
    print(f"{Style.BRIGHT}User coverage (users_count=50, min_rating=2.5):{Style.NORMAL} {get_user_coverage(top_n=top_n, users_count=50, min_rating=2.5)}")

    # Q3 - Co-clustering algorithm
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q3 - Co-clustering algorithm #############################################{Style.RESET_ALL}")
    print("Model initialization...")
    data_iterator = surprise.model_selection.KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)  # Initialize the data iterator
    param_grid = {
        "n_cltr_u": [1, 3, 5],
        "n_cltr_i": [1, 3, 5],
        "n_epochs": [10, 20, 30],
    }

    print(f"Running GridSearchCV...{Fore.WHITE}{Style.DIM}")
    grid_search = surprise.model_selection.GridSearchCV(surprise.CoClustering, param_grid=param_grid, measures=["rmse", "mae"], cv=data_iterator, n_jobs=1)
    grid_search.fit(data)  # Train the model

    # Accuracy print
    print(f"{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}RMSE:{Style.NORMAL} {round(grid_search.best_score['rmse'], 4)}")
    print(f"{Style.BRIGHT}MAE:{Style.NORMAL} {round(grid_search.best_score['mae'], 4)}")
    print(f"{Style.BRIGHT}Best params:{Style.NORMAL} {grid_search.best_params}")
    accuracy_df = pandas.DataFrame.from_dict(grid_search.cv_results)
    print(accuracy_df)

    # Cross validation
    print(f"Running the cross validation on the best model...{Fore.WHITE}{Style.DIM}")
    model_co_clustering = grid_search.best_estimator["rmse"]
    surprise.model_selection.cross_validate(model_co_clustering, data, measures=["rmse", "mae"], cv=data_iterator, n_jobs=1, verbose=True)
    print(f"{Style.RESET_ALL}")

    # Metrics
    print(f"Calculating model metrics...{Fore.WHITE}{Style.DIM}")
    data_train, data_test = surprise.model_selection.train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)  # Split the dataset
    model_co_clustering.fit(data_train)
    predictions = model_co_clustering.test(data_test)
    print(f"{Style.RESET_ALL}")

    top_n = get_top_n(predictions=predictions, n=10, min_rating=2.5)
    print(f"Built top N for each user (n=10, min_rating=2.5)")
    print(f"{Style.BRIGHT}Hit rate:{Style.NORMAL} {get_hit_rate(top_n=top_n, left_out_predictions=predictions) * 100}%")
    print(f"{Style.BRIGHT}Cumulative hit rate (min_rating=2.5):{Style.NORMAL} {get_cumulative_hit_rate(top_n=top_n, left_out_predictions=predictions, min_rating=2.5) * 100}%")
    print(f"{Style.BRIGHT}Average reciprocal hit rate:{Style.NORMAL} {get_average_reciprocal_hit_rate(top_n=top_n, left_out_predictions=predictions)}")
    print(f"{Style.BRIGHT}User coverage (users_count=50, min_rating=2.5):{Style.NORMAL} {get_user_coverage(top_n=top_n, users_count=50, min_rating=2.5)}")


if __name__ == "__main__":
    # Packages init
    colorama.init(autoreset=False)
    numpy.random.seed(RANDOM_STATE)

    # Main
    main()