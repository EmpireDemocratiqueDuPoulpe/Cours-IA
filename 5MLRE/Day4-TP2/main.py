import pandas
import colorama
from colorama import Fore, Style
import numpy
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

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


# ### Functions - Predictions ##########################################################################################
def get_similar_movies(df: pandas.DataFrame, movie_row: pandas.Series, indices: numpy.ndarray) -> pandas.DataFrame:
    movie_id = df[df["Series_Title"] == movie_row["Series_Title"]].index.tolist()[0]
    related_movies = []

    for related_movie_id in indices[movie_id][1:]:
        related_movies.append(df.loc[related_movie_id])

    return pandas.DataFrame(related_movies)


# ### Main #############################################################################################################
def main() -> None:
    data = pandas.read_csv("./data/imdb_top_1000.csv", thousands=",")

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
    data = data[["Series_Title", "Released_Year", "Certificate", "Runtime", "Genre", "IMDB_Rating", "Meta_score", "Director", "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross"]]

    print(f"{Style.BRIGHT}Filtering rows...{Style.RESET_ALL}")
    data.dropna(subset=["Gross", "Meta_score", "Certificate"], axis=0, inplace=True)

    print(f"{Style.BRIGHT}Converting the columns type...{Style.RESET_ALL}")
    data["Released_Year"] = pandas.to_numeric(data["Released_Year"], errors="coerce")
    data.dropna(subset=["Released_Year"], inplace=True)
    data["Released_Year"] = data["Released_Year"].astype(int)

    data["Certificate"] = data["Certificate"].astype("category")
    data["Runtime"] = data["Runtime"].map(lambda x: x.rstrip("min").strip()).astype(int)
    data["Gross"] = data["Gross"].astype(int)
    print(data.dtypes)

    print(f"{Style.BRIGHT}Splitting the \"Genre\" column...{Style.RESET_ALL}")
    genre_set = set()

    for genres in data["Genre"].str.split(","):
        genres = [g.strip() for g in genres]
        genre_set.update(genres)

    for genre in genre_set:
        data[genre] = 0
        data.loc[data["Genre"].str.contains(genre), genre] = 1

    data.drop("Genre", axis=1, inplace=True)
    print(data.columns)


    # Q3 - Model training
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q3 - Model training ######################################################{Style.RESET_ALL}")
    data.reset_index(drop=True, inplace=True)
    data_x = data.drop(["Series_Title"], axis=1)
    data_y = data["Series_Title"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["Released_Year", "Runtime", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross"]
    categorical_features = ["Certificate", "Director", "Star1", "Star2", "Star3", "Star4"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )
    data_x = preprocessor.fit_transform(data_x)

    model = NearestNeighbors(n_neighbors=11, metric="cosine")
    model.fit(data_x, data_y)
    distances, indices = model.kneighbors(data_x)

    # Q4 - Find close movies
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q4 - Find close movies ###################################################{Style.RESET_ALL}")
    random_movie = data.sample(n=1).iloc[0]
    related_movies = get_similar_movies(df=data, movie_row=random_movie, indices=indices)

    pandas.set_option("display.max_columns", 500)
    print(f"{Style.BRIGHT}Randomly selected movie:{Style.RESET_ALL}\n{random_movie}")
    print(f"{Style.BRIGHT}Related movies:{Style.RESET_ALL}\n{related_movies}")


if __name__ == "__main__":
    # Packages init
    colorama.init(autoreset=False)
    numpy.random.seed(RANDOM_STATE)

    # Main
    main()