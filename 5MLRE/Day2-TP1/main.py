import pandas
import colorama
from colorama import Fore, Style
import surprise
from surprise import accuracy as surp_acc
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


def get_top_n(predictions: list, n: int = 5, min_rating: float = 4.0) -> dict[str, list]:
    """ Return the top-N recommendation for each user """
    # Map the predictions to each user
    top_n = map_predictions(predictions)

    # Compute the top N of each user
    for user_id, user_ratings in top_n.items():
        # Filter the rating below the threshold
        user_ratings_filtered = list(filter(lambda x: x[1] >= min_rating, user_ratings))

        # Sort the user ratings by the estimated value
        user_ratings_filtered.sort(key=(lambda x: x[1]), reverse=True)

        # Update the top n
        top_n[user_id] = user_ratings_filtered[:n]

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
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q1 - Load datasets #######################################################")
    data = surprise.Dataset.load_builtin(name="ml-100k", prompt=True)

    # Q2 - Implement the random recommendation system
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q2 - Implement the random recommendation system ##########################")
    data_train, data_test = surprise.model_selection.train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)  # Split the dataset
    data_iterator = surprise.model_selection.LeaveOneOut(n_splits=5, random_state=RANDOM_STATE)  # Initialize the data iterator
    model_random = surprise.SVD(random_state=RANDOM_STATE, verbose=False)  # Prepare the model
    i = 0

    for data_train, data_test in data_iterator.split(data):
        print(f"\nProcessing fold {i + 1}")
        model_random.fit(data_train)  # Train the model
        predictions_random = model_random.test(data_test)  # Predict using the test dataset

        # Accuracy calculation
        surp_acc.rmse(predictions_random)
        surp_acc.mae(predictions_random)

        i += 1

    # Q3 - Implement the basic recommendation system with hyper-parameters
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q3 - Implement the basic recommendation system with hyper-parameters #####")
    data_iterator = surprise.model_selection.LeaveOneOut(n_splits=5, random_state=RANDOM_STATE)  # Initialize the data iterator
    param_grid = {
        "n_epochs": [5, 10],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6]
    }
    grid_search = surprise.model_selection.GridSearchCV(surprise.SVD, param_grid=param_grid, measures=["rmse", "mae"], cv=data_iterator, n_jobs=1)
    grid_search.fit(data)  # Train the model

    print(f"RMSE: {round(grid_search.best_score['rmse'], 4)}")
    print(f"MAE: {round(grid_search.best_score['mae'], 4)}")
    print(f"Best params: {grid_search.best_params}")
    accuracy_df = pandas.DataFrame.from_dict(grid_search.cv_results)
    print(accuracy_df)

    # Q4 - Metrics implementation
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q4 - Metrics implementation ##############################################")
    model_basic = grid_search.best_estimator["rmse"]
    model_basic.fit(data_train)
    predictions_random = model_random.test(data_test)
    predictions_basic = model_basic.test(data_test)

    print(f"\n---- Random model performance")
    top_n = get_top_n(predictions=predictions_random, n=10, min_rating=4.0)
    print(f"Top N for each user (n=10, min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{top_n}")
    print(f"Hit rate: {Fore.WHITE}{Style.BRIGHT}{get_hit_rate(top_n=top_n, left_out_predictions=predictions_random) * 100}%")
    print(f"Cumulative hit rate (min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{get_cumulative_hit_rate(top_n=top_n, left_out_predictions=predictions_random, min_rating=4.0) * 100}%")
    print(f"Average reciprocal hit rate: {Fore.WHITE}{Style.BRIGHT}{get_average_reciprocal_hit_rate(top_n=top_n, left_out_predictions=predictions_random)}")
    print(f"User coverage (users_count=50, min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{get_user_coverage(top_n=top_n, users_count=50, min_rating=4.0) * 100}%")

    print(f"\n---- Basic model performance")
    top_n = get_top_n(predictions=predictions_basic, n=10, min_rating=4.0)
    print(f"Top N for each user (n=10, min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{top_n}")
    print(f"Hit rate: {Fore.WHITE}{Style.BRIGHT}{get_hit_rate(top_n=top_n, left_out_predictions=predictions_basic) * 100}%")
    print(f"Cumulative hit rate (min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{get_cumulative_hit_rate(top_n=top_n, left_out_predictions=predictions_basic, min_rating=4.0) * 100}%")
    print(f"Average reciprocal hit rate: {Fore.WHITE}{Style.BRIGHT}{get_average_reciprocal_hit_rate(top_n=top_n, left_out_predictions=predictions_basic)}")
    print(f"User coverage (users_count=50, min_rating=4.0): {Fore.WHITE}{Style.BRIGHT}{get_user_coverage(top_n=top_n, users_count=50, min_rating=4.0) * 100}%")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()