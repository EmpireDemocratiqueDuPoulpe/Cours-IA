import pandas
from pathlib import Path
import colorama
from colorama import Fore, Style
from matplotlib import pyplot, ticker
import numpy

# ### Constants ########################################################################################################
DATASET_FOLDER = Path(__file__).resolve().parent / "data" / "movie-lens-latest-small"


# ### Main #############################################################################################################
def main() -> None:
    # Q1 - Load datasets
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q1 - Load datasets #######################################################")
    data_links = pandas.read_csv(DATASET_FOLDER / "links.csv", sep=",")
    data_movies = pandas.read_csv(DATASET_FOLDER / "movies.csv", sep=",")
    data_ratings = pandas.read_csv(DATASET_FOLDER / "ratings.csv", sep=",")
    data_tags = pandas.read_csv(DATASET_FOLDER / "tags.csv", sep=",")

    # Q2 - How many movies in this study?
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q2 - How many movies in this study? ######################################")
    movies_count = data_movies["title"].nunique(dropna=True)
    movies_duplicates_count = len(data_movies["title"]) - len(data_movies["title"].drop_duplicates())

    print(f"Loaded {movies_count} movie{'s' if (movies_count > 1) else ''}.")
    print(f"There {'are' if (movies_duplicates_count > 1) else 'is'} {movies_duplicates_count} duplicated movie{'s' if (movies_duplicates_count > 1) else ''}.")

    # Q3 - Rating matrix
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q3 - Rating matrix #######################################################")
    ratings_matrix = data_ratings.pivot(index="userId", columns="movieId", values="rating")
    matrix_empty_count = ratings_matrix.isnull().sum().sum()

    print(ratings_matrix)
    print(f"{Fore.WHITE}{Style.DIM}There {'are' if (matrix_empty_count > 1) else 'is'} {matrix_empty_count} empty value{'s' if (matrix_empty_count > 1) else ''}.")

    # Q4 - Create a movie dict
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q4 - Create a movie dict #################################################")
    movie_title_to_id = dict(zip(data_movies["title"], data_movies["movieId"]))
    movie_id_to_title = dict(zip(data_movies["movieId"], data_movies["title"]))

    # Q5 - Study the "rating" column
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q5 - Study the \"rating\" column ###########################################")
    print(data_ratings["rating"].describe())

    ax = data_ratings["rating"].plot.hist(bins=50)
    pyplot.title("Rating distribution")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # Q6 - Number of ratings per movie
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q6 - Number of ratings per movie #########################################")
    ratings_per_movie = data_ratings.groupby(["movieId"])["rating"].count().reset_index(name="ratingsCount")
    print(ratings_per_movie.describe())

    # Q7 - Movie rank based on ratings count
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q7 - Movie rank based on ratings count ###################################")
    movie_ranks_per_ratings_count = ratings_per_movie.sort_values("ratingsCount", ascending=False, inplace=False)
    movie_ranks_per_ratings_count["rank"] = range(1, 1 + len(movie_ranks_per_ratings_count))
    print(movie_ranks_per_ratings_count)

    # Q8 - Mean rating per movie
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q8 - Mean rating per movie ###############################################")
    mean_ratings_per_movie = data_ratings.groupby(["movieId"])["rating"].mean().reset_index(name="meanRating")
    print(mean_ratings_per_movie)
    print(mean_ratings_per_movie["meanRating"].describe())

    # Q9 - Movie rank based on the mean rating
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q9 - Movie rank based on the mean rating #################################")
    movie_ranks_per_mean = mean_ratings_per_movie.sort_values("meanRating", ascending=False, inplace=False)
    movie_ranks_per_mean["rank"] = range(1, 1 + len(movie_ranks_per_mean))
    print(movie_ranks_per_mean)

    # Q10.1 - Number of ratings per genre
    print(f"{Fore.GREEN}{Style.BRIGHT} #### Q10.1 - Number of ratings per genre ######################################")
    ratings_with_genre = pandas.merge(left=data_ratings, right=data_movies[["movieId", "genres"]], on="movieId", how="inner")
    ratings_with_genre = pandas.concat([ratings_with_genre, ratings_with_genre["genres"].str.get_dummies()], axis=1).drop(["genres"], axis=1)

    print(ratings_with_genre)


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()