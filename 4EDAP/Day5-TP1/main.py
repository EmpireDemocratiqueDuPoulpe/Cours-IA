import pandas
import matplotlib.pyplot as pyplot
import numpy


class TStyles:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# 1. Load the dataset
data = pandas.read_csv("./data/marvel.csv")

# 2. First look to the data
print(f"{TStyles.HEADER}{TStyles.BOLD}#### First look ##################################################{TStyles.ENDC}")
print(f"{TStyles.GREEN}## Columns{TStyles.ENDC}")
print(data.dtypes)
print(f"{TStyles.GREEN}## First rows{TStyles.ENDC}")
print(data.head(n=5))
print(data.shape)
print(f"{TStyles.GREEN}## How many missing values?{TStyles.ENDC}")
print(data.isna().sum())

# 3. Remove the column with the most of missing values
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Cleaning ####################################################{TStyles.ENDC}")
print(f"{TStyles.WARNING}Removing the \"GSM\" column...{TStyles.ENDC}")
del data["GSM"]

# 4. Analyze of the column named "ALIGN"


def bar_chart(data, column, title, ylabel):
    print(f"{TStyles.HEADER}{TStyles.BOLD}#### \"{column}\" column #####################################{TStyles.ENDC}")
    print(f"{TStyles.GREEN}## Values frequencies{TStyles.ENDC}")
    value_counts = data[column].value_counts()
    print(value_counts)

    print(f"{TStyles.BLUE}Generating a bar chart...{TStyles.ENDC}")
    fig, ax = pyplot.subplots()

    x_labels = value_counts.index.tolist()
    x_labels_pos = numpy.arange(len(x_labels))
    x_values = value_counts.tolist()
    bar_width = 0.50

    rect1 = ax.bar(x_labels_pos, x_values, width=bar_width, align="center")

    ax.set_title(title)
    ax.set_xticks(x_labels_pos, x_labels)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)

    pyplot.tight_layout()
    pyplot.show()


bar_chart(
    data,
    column="ALIGN",
    title="Répartition des personnages par BON/NEUTRE/MAUVAIS",
    ylabel="Nombre de personnage dans cette catégorie"
)

# 6. Analyze of the column named "ALIVE"
bar_chart(
    data,
    column="ALIVE",
    title="Nombre de personnages en vie/mort",
    ylabel=""
)
