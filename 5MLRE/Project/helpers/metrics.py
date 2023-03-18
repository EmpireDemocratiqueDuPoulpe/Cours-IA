# #################################################################################################################### #
#       ./helpers/metrics.py                                                                                           #
#           Functions that compute metrics to estimate the performance of the recommendation system.                   #
# #################################################################################################################### #

# Data
import pandas

# Model processing
import surprise

# Console output
from colorama import Style, Fore

# Jupyter output
from IPython.utils import io

# Misc.
from collections import defaultdict
import itertools


# ### Top-N ############################################################################################################
def map_predictions(predictions: list) -> defaultdict[int, list[list[int, float, float]]]:
    """ Map the predictions to each user. """
    user_to_predictions = defaultdict(list)
    num_impossible = 0

    for user_id, item_id, true_rating, estimated_rating, details in predictions:
        user_to_predictions[int(user_id)].append((int(item_id), float(estimated_rating), float(true_rating)))

        if details["was_impossible"]:
            num_impossible += 1

    if num_impossible > 0:
        num_predictions = len(predictions)
        print(f"{Fore.YELLOW}Warning: {num_impossible}/{num_predictions} ({(num_impossible * 100) / num_predictions:.6f}%) predictions were impossible! {Fore.RESET}")  # noqa: E501

    return user_to_predictions


def get_top_n(predictions: list, n: int = 10, min_rating: float = 4.0, verbose: bool = False) -> dict[int, list[list[int, float, float]]]:  # noqa: E501
    """ Returns the top-N recommendation for each user. """
    top_n = map_predictions(predictions)

    # Compute the top-N for each user
    for user_id, ratings in top_n.items():
        ratings_sorted = sorted(ratings, key=(lambda x: x[1]), reverse=True)  # Sort the ratings list by the estimated rating of each item  # noqa: E501
        ratings_shortened = ratings_sorted[:n]  # Cut down the top
        top_n[user_id] = list(filter((lambda x: x[1] >= min_rating), ratings_shortened))  # Filter the ratings below the threshold  # noqa: E501

    if verbose:
        print(f"Built top-N for each user (n={n}, min_rating={min_rating})")

    return dict(top_n)


def is_in_top_n(top_n: dict[int, list], user_id: int | str, item_id: int | str) -> bool:
    """ Returns true if the `item_id` is in the `top_n` of `user_id`. """
    for top_item_id, _, _ in top_n[int(user_id)]:
        if top_item_id == int(item_id):
            return True

    return False


# ### Hit rates ########################################################################################################
def get_hit_rate(top_n: dict[int, list], left_out_predictions: list, auto_print: bool = False) -> float:
    """ Compute the overall hit rate. """
    hits = 0
    total = 0

    # Compute the hit rate
    for user_id, item_id, _, _, _ in left_out_predictions:
        hit = is_in_top_n(top_n=top_n, user_id=user_id, item_id=item_id)

        total += 1
        if hit:
            hits += 1

    hit_rate = total and (hits / total) or 0  # Prevent "division by zero" error

    # Print to console
    if auto_print:
        print(f"{Style.BRIGHT}Hit rate:{Style.NORMAL} {(hit_rate * 100):.6f}%")

    return hit_rate


def get_rating_hit_rate(top_n: dict[int, list], left_out_predictions: list, auto_print: bool = False) -> dict[str, float]:  # noqa: E501
    """ Compute the overall hit rate per rating. """
    hits = defaultdict(float)
    total = defaultdict(float)

    # Compute the hit rate
    for user_id, item_id, true_rating, _, _ in left_out_predictions:
        hit = is_in_top_n(top_n=top_n, user_id=user_id, item_id=item_id)

        total[true_rating] += 1
        if hit:
            hits[true_rating] += 1

    hit_rate = {}

    if auto_print:
        print(f"{Style.BRIGHT}Hit rate per rating value:{Style.NORMAL}")
        print("Rating\tHit rate")

    for rating in sorted(hits.keys()):
        hit_rate[rating] = total[rating] and (hits[rating] / total[rating]) or 0  # Prevent "division by zero" error

        if auto_print:
            print(f"{rating}\t{(hit_rate[rating] * 100):.6f}%")

    return hit_rate


def get_cumulative_hit_rate(top_n: dict[int, list], left_out_predictions: list, min_rating: float = 4.0, auto_print: bool = False) -> float:  # noqa: E501
    """ Compute the cumulative hit rate. """
    hits = 0
    total = 0

    # Compute the hit rate
    for user_id, item_id, true_rating, _, _ in left_out_predictions:
        if true_rating >= min_rating:
            hit = is_in_top_n(top_n=top_n, user_id=user_id, item_id=item_id)

            total += 1
            if hit:
                hits += 1

    hit_rate = total and (hits / total) or 0  # Prevent "division by zero" error

    # Print to console
    if auto_print:
        print(f"{Style.BRIGHT}Cumulative hit rate (min_rating={min_rating}):{Style.NORMAL} {(hit_rate * 100):.6f}%")

    return hit_rate


def get_average_reciprocal_hit_rank(top_n: dict[int, list], left_out_predictions: list, auto_print: bool = False) -> float:  # noqa: E501
    """ Compute average reciprocal hit rank (ARHR). """
    summation = 0
    total = 0

    # Compute the hit rate
    for user_id, item_id, _, _, _ in left_out_predictions:
        # Get the hit rank of the item. The lower, the better
        hit_rank = 0
        rank = 0

        for top_item_id, _, _ in top_n[int(user_id)]:
            rank += 1

            if top_item_id == int(item_id):
                hit_rank = rank
                break

        total += 1
        if hit_rank > 0:
            summation += 1.0 / hit_rank

    hit_rank = total and (summation / total) or 0  # Prevent "division by zero" error

    # Print to console
    if auto_print:
        print(f"{Style.BRIGHT}Average reciprocal hit rank:{Style.NORMAL} {hit_rank}")

    return hit_rank


# ### Coverages ########################################################################################################
def get_user_coverage(top_n: dict[int, list], num_users: int, min_rating: float = 4.0, auto_print: bool = False) -> float:  # noqa: E501
    """ Compute the user coverage. """
    hits = 0

    for user_id in top_n.keys():
        hit = False  # Is there an item in the top-N with an estimated rating above `min_rating`?

        for _, top_estimated_rating, _ in top_n[user_id]:
            if top_estimated_rating >= min_rating:
                hit = True
                break

        if hit:
            hits += 1

    user_coverage = num_users and (hits / num_users) or 0  # Prevent "division by zero" error

    # Print to console
    if auto_print:
        print(f"{Style.BRIGHT}User coverage (num_users={num_users}, min_rating={min_rating}):{Style.NORMAL} {(user_coverage * 100):.6f}%")  # noqa: E501

    return user_coverage


# ### Diversity ########################################################################################################
def get_diversity(top_n: dict[int, list], model: surprise.prediction_algorithms.algo_base.AlgoBase, auto_print: bool = False) -> float:  # noqa: E501
    """ Compute the diversity. """
    n = 0
    total = 0
    
    with io.capture_output():
        similarities_matrix = model.compute_similarities()

    for user_id in top_n.keys():
        pairs = itertools.combinations(iterable=top_n[user_id], r=2)

        for pair in pairs:
            item1 = pair[0][0]
            item2 = pair[1][0]
            inner_id1 = model.trainset.to_inner_iid(str(item1))
            inner_id2 = model.trainset.to_inner_iid(str(item2))

            similarity = similarities_matrix[inner_id1][inner_id2]
            total += similarity
            n += 1

        S = n and (total / n) or 0  # Prevent "division by zero" error
        S_minus_one = 1 - S

        if auto_print:
            print(f"{Style.BRIGHT}Diversity:{Style.NORMAL} {S_minus_one:.6f}")

        return S_minus_one


# ### Novelty ##########################################################################################################
def get_novelty(top_n: dict[int, list], rankings: dict[int, int], auto_print: bool = False) -> float:
    """ Compute novelty. """
    n = 0
    total = 0

    for user_id in top_n.keys():
        for top_item_id, _, _ in top_n[user_id]:
            rank = rankings[top_item_id]

            total += rank
            n += 1

    novelty = n and (total / n) or 0  # Prevent "division by zero" error

    if auto_print:
        print(f"{Style.BRIGHT}Novelty:{Style.NORMAL} {novelty:.6f}")

    return novelty
