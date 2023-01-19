# #################################################################################################################### #
#       search.py                                                                                                      #
#           Model selection stuff.                                                                                     #
# #################################################################################################################### #

from colorama import Fore, Style


def best_model(model, search, x_train, y_train, x_test, y_test, train_score=None, test_score=None):
    train_score = train_score if train_score else round(model.score(x_train, y_train) * 100, 2)
    test_score = test_score if test_score else round(model.score(x_test, y_test) * 100, 2)

    search.fit(x_train, y_train)
    print(f"Best params for model: {Fore.LIGHTGREEN_EX}{search.best_params_}")
    new_model = search.best_estimator_

    new_train_score = round(new_model.score(x_train, y_train) * 100, 2)
    new_test_score = round(new_model.score(x_test, y_test) * 100, 2)
    delta_train = round((new_train_score - train_score) / 100, 2)
    delta_test = round((new_test_score - test_score) / 100, 2)

    print((
        f"New model score:\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ train → {new_train_score}% ({delta_to_str(delta_train)}{Fore.WHITE}{Style.DIM})\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ test → {new_test_score}% ({delta_to_str(delta_test)}{Fore.WHITE}{Style.DIM})"
    ))

    if new_train_score >= train_score and new_test_score >= test_score:
        return new_model
    elif new_train_score < train_score and new_test_score < test_score:
        return model
    else:
        if new_train_score < new_test_score:
            resolve = {"lower_score": new_train_score, "prev_score": train_score}
        else:
            resolve = {"lower_score": new_test_score, "prev_score": test_score}

        return new_model if ((resolve["lower_score"] - resolve["prev_score"]) >= 0) else model


def delta_to_str(delta):
    color = Fore.RED
    sign = ""

    if delta > -1:
        color = Fore.GREEN
        sign = "+"
        delta = abs(delta)

    return f"{color}{sign}{delta}%{Fore.RESET}"
