# #################################################################################################################### #
#       ./ml.py                                                                                                #
#           Machine Learning functions.                                                                                #
# #################################################################################################################### #

# OS and filesystem
from pathlib import Path
import pickle
from timeit import default_timer as timer
from datetime import timedelta
import random

# Math
import numpy

# Model processing
import surprise
from surprise import accuracy as surp_acc

# Console output
from colorama import Fore, Style

# Jupyter output
from IPython.utils import io

# Typing
from typing import Type

# Local files
from . import metrics


# ### Seeds ############################################################################################################
def reset_random_seed(seed: int | None = 0) -> None:
    """ Reset the random seed to have reproducible experiments. """
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)


# ### Model training ###################################################################################################
def run_model(name: str,
              model: Type[surprise.prediction_algorithms.algo_base.AlgoBase],
              dataset: surprise.dataset.Dataset,
              hyper_params: dict | None = None,
              measures: list[str] | None = None,
              measure_key: str = "rmse",
              models_folder: Path | None = None,
              seed: int | None = None
    ) -> tuple[surprise.prediction_algorithms.algo_base.AlgoBase, dict, dict[int, list]]:
    """
    Train the model, using GridSearch if possible, and send the models metrics to the console.
    Returns a tuple containing the best model, the best hyper-parameters dict and the predicted top-N.
    """
    # Initialize the function parameters
    if measures is None:
        measures = ["rmse", "mae"]

    # Initialize the model processing
    run_start_time = timer()
    print(f"{Fore.GREEN}Testing \"{name}\".{Fore.RESET}")

    # Train the model
    if hyper_params is not None:  # If available, search the best estimator with GridSearch
        print(f"Running GridSearchCV...{Fore.WHITE}{Style.DIM}")
        reset_random_seed(seed)

        grid_search_start_time = timer()
        with io.capture_output():
            grid_search = surprise.model_selection.GridSearchCV(
                algo_class=model,
                param_grid=hyper_params,
                measures=measures,
                cv=surprise.model_selection.KFold(n_splits=10, random_state=seed, shuffle=True),
                refit=False,
                n_jobs=1,
                joblib_verbose=0
            )
            grid_search.fit(dataset)

        best_model = grid_search.best_estimator[measure_key]
        best_params = grid_search.best_params[measure_key]
        grid_search_end_time = timer()
    else:
        reset_random_seed(seed)
        best_model = model()
        best_params = {}
        grid_search_start_time = grid_search_end_time = None

    # Accuracy calculation
    top_n, model_start_time, model_end_time = compute_metrics(model=best_model, dataset=dataset, best_params=best_params, seed=seed)

    # Save the model to the disk
    if models_folder:
        surprise.dump.dump(file_name=str(models_folder / name), algo=best_model)

        with (models_folder / f"{name}_topN.pkl").open(mode="wb") as dump_file:
            pickle.dump(obj=top_n, file=dump_file)

    # Final output
    run_end_time = timer()
    run_elapsed_time = timedelta(seconds=run_end_time - run_start_time)
    grid_search_elapsed_time = timedelta(seconds=grid_search_end_time - grid_search_start_time) if grid_search_start_time is not None else None
    model_elapsed_time = timedelta(seconds=model_end_time - model_start_time) if model_start_time is not None else None
    print((
        f"\nTesting of the \"{name}\" model successfully completed in {run_elapsed_time}."
        f"\nGrid search: {'N/A' if grid_search_elapsed_time is None else grid_search_elapsed_time}"
        f"\nTraining and testing: {'N/A' if model_elapsed_time is None else model_elapsed_time}"
    ))

    return best_model, best_params, top_n


# ### Model training ###################################################################################################
def compute_metrics(
        model: surprise.prediction_algorithms.algo_base.AlgoBase,
        dataset: surprise.dataset.Dataset,
        best_params: dict | None = None,
        seed: int | None = None
    ) -> tuple[dict[int, list], float, float]:
    """
    Compute the metrics of the model (RMSE, MAE, Hit-Rate, ...).
    Returns a tuple containing the predicted top-N and the start and the end time of the training.
    """
    top_n = model_start_time = model_end_time = None
    LOOCV = surprise.model_selection.LeaveOneOut(n_splits=1, min_n_ratings=1, random_state=seed)

    reset_random_seed(seed)
    for data_train_LOOCV, data_test_LOOCV in LOOCV.split(dataset):
        # Train the model with Leave One Out
        model_start_time = timer()
        model.fit(data_train_LOOCV)
        train_prediction = model.test(data_train_LOOCV.build_testset())
        left_out_predictions = test_prediction = model.test(data_test_LOOCV)
        all_predictions = model.test(data_train_LOOCV.build_anti_testset())
        model_end_time = timer()

        # Show the best params, the RMSE and the MAE
        print(f"{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}Best params:{Style.NORMAL} {Style.DIM}{Fore.WHITE}{best_params if best_params is not None else ({})}{Style.RESET_ALL}")
        print((
            f"{Style.BRIGHT}RMSE:{Style.NORMAL} "
            f"[ train = {surp_acc.rmse(train_prediction, verbose=False):.4f} | "
            f"test = {surp_acc.rmse(test_prediction, verbose=False):.4f} ]"
        ))
        print((
            f"{Style.BRIGHT}MAE:{Style.NORMAL} "
            f"[ train = {surp_acc.mae(train_prediction, verbose=False):.4f} | "
            f"test = {surp_acc.mae(test_prediction, verbose=False):.4f} ]"
        ))
        print("")

        # Compute the top-N of each user
        top_n = metrics.get_top_n(predictions=all_predictions, n=10, min_rating=4.0, verbose=True)
        metrics.get_hit_rate(top_n=top_n, left_out_predictions=left_out_predictions, auto_print=True)
        metrics.get_rating_hit_rate(top_n=top_n, left_out_predictions=left_out_predictions, auto_print=True)
        metrics.get_cumulative_hit_rate(top_n=top_n, left_out_predictions=left_out_predictions, min_rating=4.0, auto_print=True)
        metrics.get_average_reciprocal_hit_rank(top_n=top_n, left_out_predictions=left_out_predictions, auto_print=True)
        metrics.get_user_coverage(top_n=top_n, num_users=data_train_LOOCV.n_users, min_rating=4.0, auto_print=True)
        metrics.get_diversity(top_n=top_n, model=model, auto_print=True)

    return top_n, model_start_time, model_end_time
