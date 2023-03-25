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
import re

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
from typing import Type, Literal

# Local files
from . import metrics


# ### ModelEvaluator ###################################################################################################
class ModelEvaluator:
    # === Initialization ===============================================================================================
    def __init__(self,
                 dataset: surprise.dataset.DatasetAutoFolds,
                 rankings: dict[int, int],
                 models_folder: Path,
                 seed: int | None = None
                 ) -> None:
        self._dataset = dataset  # Full dataset without folds
        self._rankings = rankings  # Items ranking
        self._seed = seed  # Random seed to use to have reproducible experiments
        self._use_seed = True if self._seed is not None else False

        # Build the various set
        self._train_set = self._test_set = self._train_LOOCV = self._test_LOOCV = self._anti_test_LOOCV = \
            self._full_train_set = self._full_anti_test_set = self._similarities_model = None
        self.construct_sets(with_similarities=True)

        # Models training
        self._models_folder = models_folder
        self._models: dict[str, Model] = {}  # Save the models params and metrics in this dict

    def construct_sets(self, with_similarities: bool = True):
        """ Prepare the various set used in the model evaluation. Use a lot of RAM. """
        # Training and testing sets to compute basic metrics [RMSE, MAE, ...]
        print(f"Constructing sets. This can take a while...{Fore.WHITE}{Style.DIM}")

        print(f"{Fore.WHITE}{Style.DIM}   > Building train/test sets...{Fore.WHITE}{Style.DIM}")
        self._reset_random_seed()
        self._train_set, self._test_set = surprise.model_selection.train_test_split(
            data=self._dataset,
            test_size=0.2,
            shuffle=True,
            random_state=self._get_seed()
        )

        # LeaveOneOut sets for top-N and hit-rate metrics
        print(f"{Fore.WHITE}{Style.DIM}   > Building LeaveOneOut sets...{Fore.WHITE}{Style.DIM}")
        self._reset_random_seed()
        # LOOCV = surprise.model_selection.LeaveOneOut(n_splits=1, random_state=self._get_seed())
        LOOCV = surprise.model_selection.LeaveOneOut(n_splits=1, min_n_ratings=1, random_state=self._get_seed())

        for data_train_LOOCV, data_test_LOOCV in LOOCV.split(data=self._dataset):
            self._train_LOOCV = data_train_LOOCV
            self._test_LOOCV = data_test_LOOCV
            del data_train_LOOCV, data_test_LOOCV
        self._anti_test_LOOCV = self._train_LOOCV.build_anti_testset()

        # Full sets
        print(f"{Fore.WHITE}{Style.DIM}   > Building full sets...{Fore.WHITE}{Style.DIM}")
        self._full_train_set = self._dataset.build_full_trainset()
        self._full_anti_test_set = self._full_train_set.build_anti_testset()

        # Prepare the model that compute the similarities matrix for the "diversity" metric
        if with_similarities:
            print(f"{Fore.WHITE}{Style.DIM}   > Preparing the similarities model...{Fore.WHITE}{Style.DIM}")
            self._similarities_model = surprise.KNNBaseline(sim_options={"name": "cosine", "user_based": False})
            self._similarities_model.fit(self._full_train_set)

        print(f"{Style.RESET_ALL}")

    # === Setters ======================================================================================================
    def set_seed(self, seed: int, enable: bool = True) -> None:
        """ Update the seed. """
        self._seed = seed

        if enable:
            self.enable_seeding()
        else:
            self.disable_seeding()

    def enable_seeding(self) -> None:
        """ Enable the use of the seed. """
        self._use_seed = True

    def disable_seeding(self) -> None:
        """ Disable the use of the seed. """
        self._use_seed = False

    # === Getters ======================================================================================================
    def is_seeding(self) -> bool:
        """ Returns True if the seed is used. """
        return self._use_seed

    def _get_seed(self) -> int | None:
        """ Returns the seed if its enabled, None otherwise. """
        return self._seed if self._use_seed else None

    # === Models training ==============================================================================================
    def _add_model(self, name: str, override: bool = False) -> None:
        """ Initialize the model in the models' dict. """
        # Check that the name doesn't override another model
        if (name in self._models.keys()) and not override:
            raise ValueError((
                f"The model name \"{name}\" is already used! Change it or pass `override=True` to this function."
            ))

        # Add the model
        self._models[name] = Model(name=name)

    def run_model(self,
                  name: str,
                  model: Type[surprise.prediction_algorithms.algo_base.AlgoBase],
                  hyper_params: dict | None = None,
                  measure_key: Literal["rmse", "mae"] = "rmse",
                  override: bool = False
                  ) -> None:
        """ Run the training of a model and print its metrics. """
        # Initialize the model processing
        self._add_model(name=name, override=override)
        self._models[name].save_timing(category="run", when="start")
        print(f"{Fore.GREEN}Testing \"{self._models[name].get_name()}\".{Fore.RESET}")

        # Train the model
        self._reset_random_seed()

        if hyper_params is not None:  # If available, search the best estimator with GridSearch
            print(f"Running GridSearchCV...{Fore.WHITE}{Style.DIM}")
            self._models[name].save_timing(category="grid_search", when="start")

            with io.capture_output():
                grid_search = surprise.model_selection.GridSearchCV(
                    algo_class=model,
                    param_grid=hyper_params,
                    measures=["rmse", "mae"],
                    cv=surprise.model_selection.KFold(n_splits=10, shuffle=True, random_state=self._get_seed()),
                    refit=False,
                    n_jobs=1,
                    joblib_verbose=0
                )
                grid_search.fit(self._dataset)

            best_model = grid_search.best_estimator[measure_key]
            best_params = grid_search.best_params[measure_key]
            self._models[name].save_timing(category="grid_search", when="end")
        else:  # Get the model without GridSearch
            best_model = model()
            best_params = {}

        # Save the best estimator and its params
        self._models[name].save_best(estimator=best_model, params=best_params, folder=self._models_folder)

        # Accuracy calculation
        self._compute_metrics(name=name)

    def _compute_metrics(self, name: str) -> None:
        """ Compute and display the model metrics. """
        print(f"Computing metrics...{Fore.WHITE}{Style.DIM}")
        estimator = self._models[name].get_best_estimator()

        # Fit the model on the basic train/test set to get the accuracy metrics
        print(f"{Fore.WHITE}{Style.DIM}Calculating the accuracy (RMSE, MAE)...{Fore.WHITE}{Style.DIM}")
        self._models[name].save_timing(category="training", when="start")
        estimator.fit(self._train_set)
        predictions = estimator.test(self._test_set)
        self._models[name].save_timing(category="training", when="end")

        self._models[name].set_rmse(rmse=surp_acc.rmse(predictions, verbose=False))
        self._models[name].set_mae(mae=surp_acc.mae(predictions, verbose=False))

        # Fit the model on the LeaveOneOut sets to build the top-N
        print(f"{Fore.WHITE}{Style.DIM}Building the top-N...{Fore.WHITE}{Style.DIM}")
        self._models[name].save_timing(category="top_n_building", when="start")

        print(f"{Fore.WHITE}{Style.DIM}   > Fitting on the LOOCV...{Fore.WHITE}{Style.DIM}")
        estimator.fit(self._train_LOOCV)
        predictions_LOOCV = estimator.test(self._test_LOOCV)  # Left-out predictions
        all_predictions_LOOCV = estimator.test(self._anti_test_LOOCV)  # All predictions

        print(f"{Fore.WHITE}{Style.DIM}   > Fitting on the full set...{Fore.WHITE}{Style.DIM}")
        estimator.fit(self._full_train_set)
        all_predictions_full = estimator.test(self._full_anti_test_set)  # All predictions

        top_n_LOOCV_path = self._models[name].save_top_n(
            top_n=metrics.get_top_n(predictions=all_predictions_LOOCV, n=10, min_rating=3.0, verbose=True),
            folder=self._models_folder,
            suffix="topN-LOOCV"
        )
        top_n_FULL_path = self._models[name].save_top_n(
            top_n=metrics.get_top_n(predictions=all_predictions_full, n=10, min_rating=3.0, verbose=True),
            folder=self._models_folder,
            suffix="topN-full"
        )
        self._models[name].save_timing(category="top_n_building", when="end")

        # Show metrics
        self._show_metrics(
            name=name,
            left_out_predictions=predictions_LOOCV,
            top_n_LOOCV_path=top_n_LOOCV_path,
            top_n_FULL_path=top_n_FULL_path
        )

    def _show_metrics(self, name: str, left_out_predictions: list, top_n_LOOCV_path: Path, top_n_FULL_path: Path) -> None:  # noqa: E501
        """ Shows the previously computed metrics. This is done is another function to make it more readable. """
        # Show the best params, the RMSE and the MAE
        best_params = self._models[name].get_best_params()

        print(f"{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}Best params:{Style.NORMAL} {Style.DIM}{Fore.WHITE}{best_params if best_params is not None else ({})}{Style.RESET_ALL}")  # noqa: E501
        print(f"{Style.BRIGHT}RMSE:{Style.NORMAL} {self._models[name].get_rmse():.6f}")
        print(f"{Style.BRIGHT}MAE:{Style.NORMAL} {self._models[name].get_mae():.6f}")
        print("")

        # Shows the hit rates
        top_n_LOOCV = self._models[name].load_top_n(filepath=top_n_LOOCV_path)
        metrics.get_hit_rate(top_n=top_n_LOOCV, left_out_predictions=left_out_predictions, auto_print=True)
        metrics.get_rating_hit_rate(top_n=top_n_LOOCV, left_out_predictions=left_out_predictions, auto_print=True)
        metrics.get_cumulative_hit_rate(top_n=top_n_LOOCV, left_out_predictions=left_out_predictions, min_rating=3.0, auto_print=True)  # noqa: E501
        metrics.get_average_reciprocal_hit_rank(top_n=top_n_LOOCV, left_out_predictions=left_out_predictions, auto_print=True)  # noqa: E501
        metrics.get_user_coverage(top_n=top_n_LOOCV, num_users=self._train_LOOCV.n_users, min_rating=3.0, auto_print=True)  # noqa: E501
        del top_n_LOOCV

        # Shows the diversity and the novelty
        top_n_FULL = self._models[name].load_top_n(filepath=top_n_FULL_path)
        metrics.get_diversity(top_n=top_n_FULL, model=self._similarities_model, auto_print=True)
        metrics.get_novelty(top_n=top_n_FULL, rankings=self._rankings, auto_print=True)
        del top_n_FULL

        # Shows the computation time
        self._models[name].save_timing(category="run", when="end")
        self._models[name].print_timings()

    # === Miscellaneous ================================================================================================
    def _reset_random_seed(self) -> None:
        """ Reset the random seed to have reproducible experiments. """
        if self._use_seed is not None:
            random.seed(self._seed)
            numpy.random.seed(self._seed)


# ### Model ############################################################################################################
class Model:
    # === Initialization ===============================================================================================
    def __init__(self, name: str) -> None:
        self._name = name  # Model name
        self._best_estimator = None  # Best estimator
        self._best_params = None  # Best hyper-parameters
        self._timings = {  # Timing
            "run": {"name": None, "start": None, "end": None},
            "grid_search": {"name": "Grid search", "start": None, "end": None},
            "training": {"name": "Training and testing", "start": None, "end": None},
            "top_n_building": {"name": "Top-N building", "start": None, "end": None},
        }

        # Metrics
        self._rmse = None
        self._mae = None

    # === Setters ======================================================================================================
    def save_timing(self,
                    category: Literal["run", "grid_search", "training", "top_n_building"],
                    when: Literal["start", "end"]
                    ) -> None:
        """ Save a timing. """
        self._timings[category][when] = timer()

    def set_rmse(self, rmse: float) -> None:
        """ Update the RMSE. """
        self._rmse = rmse

    def set_mae(self, mae: float) -> None:
        """ Update the MAE. """
        self._mae = mae

    # === Getters ======================================================================================================
    def get_name(self, sanitize: bool = False) -> str:
        """ Returns the model name. """
        if sanitize:
            name_sanitized = self._name.strip().replace(" ", "_")
            return re.sub(r"(?u)[^-\w.]", "", name_sanitized)[:200]

        return self._name

    def get_best_estimator(self) -> surprise.prediction_algorithms.algo_base.AlgoBase | None:
        """ Returns the best estimator. """
        self._alert_model_state()
        return self._best_estimator

    def get_best_params(self) -> dict | None:
        """ Returns the model best hyper-parameters. """
        self._alert_model_state()
        return self._best_params

    def get_rmse(self) -> float | None:
        """ Returns the model RMSE. """
        self._alert_model_state()
        return self._rmse

    def get_mae(self) -> float | None:
        """ Returns the model MAE. """
        self._alert_model_state()
        return self._mae

    # === GridSearchCV =================================================================================================
    def save_best(self,
                  estimator: surprise.prediction_algorithms.algo_base.AlgoBase,
                  params: dict,
                  folder: Path | None = None
                  ) -> surprise.prediction_algorithms.algo_base.AlgoBase:
        """ Saves the best estimator and its params. Saves the model to the disk if `folder` is not None.
            Returns a new instance of this estimator. """
        self._best_params = params
        self._best_estimator = estimator.__class__(**self._best_params)

        if (folder is not None) and folder.is_dir():  # Save to the disk
            surprise.dump.dump(file_name=str(folder / self.get_name(sanitize=True)), algo=self._best_estimator)

        return self._best_estimator

    def save_top_n(self, top_n: dict, folder: Path, suffix: str = "topN") -> Path:
        """ Saves the top-N to the disk and return the full path. """
        dump_path = (folder / f"{self.get_name(sanitize=True)}__{suffix}.pkl")

        with dump_path.open(mode="wb") as dump_file:
            pickle.dump(obj=top_n, file=dump_file)

        return dump_path

    @staticmethod
    def load_top_n(filepath: Path) -> dict:
        """ Loads the top-N from the disk and return it. """
        if not filepath.is_file():
            raise ValueError(f"Incorrect file path! There is no dump file located at \"{filepath}\".")

        with filepath.open(mode="rb") as dump_file:
            top_n = pickle.load(file=dump_file)

        return top_n

    # === Miscellaneous ================================================================================================
    def print_timings(self) -> None:
        """ Prints the timing to the console. """
        def compute_timing(timing_obj: dict) -> timedelta | None:
            if (timing_obj["start"] is not None) and (timing_obj["end"] is not None):
                return timedelta(seconds=(timing_obj["end"] - timing_obj["start"]))
            else:
                return None

        # Print the first message
        run_elapsed_time = compute_timing(self._timings["run"])
        run_elapsed_time = f" in {run_elapsed_time}" if (run_elapsed_time is not None) else ""

        print(f"\n{Style.RESET_ALL}Testing of the \"{self.get_name()}\" model successfully completed{run_elapsed_time}.")  # noqa: E501

        # Print the sub-timings
        for key, timing in self._timings.items():
            if key == "run":
                continue
        
            name = timing["name"] if timing["name"] is not None else key
            elapsed_time = compute_timing(timing)

            print(f"{name}: {'N/A' if elapsed_time is None else elapsed_time}")

    def _alert_model_state(self, disable: bool = False) -> None:
        """ Prints a warning message if the user access a value before the model instantiation. """
        if (self._best_estimator is None) and not disable:
            print((
                f"{Fore.YELLOW} Warning: the model wasn't fitted. "
                "The value may not reflect the reality of the situation."
            ))
