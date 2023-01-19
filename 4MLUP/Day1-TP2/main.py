from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import numpy
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data_x, data_y = load_digits(return_X_y=True)

    # First look
    mlsp.misc.print_title("First look")
    print((
        f"Shape:{Style.DIM}{Fore.WHITE}"
        f" (x: {Fore.LIGHTGREEN_EX}{data_x.shape}{Fore.WHITE}"
        f" | y: {Fore.LIGHTGREEN_EX}{data_y.shape}{Fore.WHITE})"
    ))
    print(f"x:{Style.DIM}{Fore.WHITE}\n{data_x}")
    print(f"y:{Style.DIM}{Fore.WHITE} {data_y}")

    # Reshape to an image
    mlsp.misc.print_title("Show one of the digits")
    # pyplot.imshow(numpy.reshape(data_x, (192, 599)))
    # pyplot.title("Le jean (la réinvention)")
    # pyplot.axis(False)
    # pyplot.show()

    random_digit = data_x[numpy.random.randint(0, (len(data_x) - 1))]
    pyplot.imshow(numpy.reshape(random_digit, (8, 8)))
    pyplot.title("A randomly picked digit")
    pyplot.show()

    # Split data to train/test
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)

    # Model
    mlsp.misc.print_title("Model")
    classifier = KMeans(random_state=42)
    visualizer = KElbowVisualizer(classifier, k=(2, 100))

    visualizer.fit(data_x)
    visualizer.show()

    print(f"{Fore.YELLOW}Using a KMean model with n_clusters={visualizer.elbow_value_}...")
    classifier = KMeans(n_clusters=visualizer.elbow_value_, random_state=42)
    model = LogisticRegression(solver="lbfgs", multi_class="ovr", max_iter=5000, random_state=42)

    processor = Pipeline(steps=[
        ("classifier", classifier),
        ("model", model)
    ])

    model, scores = mlsp.models.common.process_model(
        processor,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        verbose=True
    )

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of digits dataset in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
