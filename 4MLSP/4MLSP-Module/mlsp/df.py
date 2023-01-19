# #################################################################################################################### #
#       df.py                                                                                                          #
#           Functions used on pandas dataframe.                                                                        #
# #################################################################################################################### #

from colorama import Fore, Style
from sklearn.model_selection import train_test_split


def first_look(df):
    print(f"Shape: {Fore.LIGHTGREEN_EX}{df.shape}")
    print(df.head())
    print(df.dtypes)


def missing_values(df, keep_zeros=True):
    data_count = df.shape[0] * df.shape[1]
    missing = missing_df = df.isna().sum()

    if not keep_zeros:
        missing_df = missing_df[missing_df > 0]

    missing_df = missing_df.sort_values(ascending=False).apply(lambda m: f"{m} ({round((m * 100) / df.shape[0], 2)}%)")

    print((
        f"Missing values: {Fore.LIGHTGREEN_EX}{round((missing.sum() / data_count) * 100, 2)}%\n"
        f"{Fore.WHITE}{Style.DIM}{missing_df}"
    ))


def split_train_test(df, y_label=None, test_size=0.20):
    if y_label is not None:
        data_x = df.drop([y_label], axis=1)
        data_y = df[y_label]

        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size)
        return x_train, x_test, y_train, y_test
    else:
        x_train, x_test = train_test_split(df, test_size=test_size)
        return x_train, x_test
