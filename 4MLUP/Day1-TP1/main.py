from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
import numpy
from matplotlib import pyplot
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/USArrests.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data)

    # Fix of state column
    mlsp.misc.print_title("Fix of state column")
    data.rename({"Unnamed: 0": "State"}, axis=1, inplace=True)
    print(f"{Style.DIM}{Fore.WHITE}{data.columns}")

    # Study
    mlsp.misc.print_title("Study")

    # Add "Total Arrests" column
    data["Total Arrests"] = data[["Murder", "Assault", "Rape"]].sum(axis=1)
    data_sorted = data.sort_values(by="Total Arrests", ascending=True)

    # Repartition per state and per crimes
    repart_labels = data_sorted["State"]
    repart_murder = data_sorted["Murder"]
    repart_assault = data_sorted["Assault"]
    repart_rape = data_sorted["Rape"]
    bar_width = 0.50

    repart_fix, repart_ax = pyplot.subplots()

    repart_ax.barh(repart_labels, repart_murder, bar_width, label="Murder")
    repart_ax.barh(repart_labels, repart_assault, bar_width, left=repart_murder, label="Assault")
    repart_ax.barh(repart_labels, repart_rape, bar_width, left=(repart_murder + repart_assault), label="Rape")

    repart_ax.set_xlabel("Crimes per 100 000 residents")
    repart_ax.tick_params(axis="y", which="both", labelsize=6)
    repart_ax.set_title("Repartition per state and per crimes")
    repart_ax.legend()

    pyplot.tight_layout()
    pyplot.show()

    # Transform values
    mlsp.misc.print_title("Transform values")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    numeric_features = ["Murder", "Assault", "Rape"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features)
        ]
    )

    data_scaled = preprocessor.fit_transform(data)

    # KMean clustering
    mlsp.misc.print_title("KMean clustering")
    mlsp.misc.print_title("> Find the best n_cluster (Elbow and Silhouette method)", char="~")
    k_range = range(1, 20)
    model_sse = []
    model_silhouette = []

    for num_k in k_range:
        model = KMeans(init="random", n_clusters=num_k, n_init=10, max_iter=300, random_state=7)
        model.fit(data_scaled)

        model_sse.append(model.inertia_)

        if num_k >= 2:
            model_silhouette.append(silhouette_score(data_scaled, model.labels_))

    pyplot.plot(k_range, model_sse)
    pyplot.xticks(k_range)
    pyplot.xlabel("Number of clusters")
    pyplot.ylabel("Inertia")
    pyplot.title("KMean clustering (Elbow method)")
    pyplot.show()

    silhouette_range = [num_k for num_k in list(k_range) if num_k >= 2]
    pyplot.plot(silhouette_range, model_silhouette)
    pyplot.xticks(silhouette_range)
    pyplot.xlabel("Number of clusters")
    pyplot.ylabel("Silhouette Coefficient")
    pyplot.title("KMean clustering (Silhouette method)")
    pyplot.show()

    mlsp.misc.print_title("> KMean with n_cluster=4", char="~")
    data_reduced = pandas.DataFrame(PCA(n_components=2).fit_transform(data[numeric_features]), columns=["pca1", "pca2"])
    model = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300, random_state=7)

    data["Cluster KMean"] = model.fit_predict(data_reduced)

    pyplot.scatter(data_reduced["pca1"], data_reduced["pca2"], c=data["Cluster KMean"], s=50)
    pyplot.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", s=200, alpha=0.5)
    pyplot.title("KMean with n_cluster=4")
    pyplot.show()

    # Agglomerative clustering
    mlsp.misc.print_title("Agglomerative clustering")
    mlsp.misc.print_title("> Find the best n_cluster (Silhouette method)", char="~")
    k_range = range(2, 20)
    model_silhouette = []

    for num_k in k_range:
        model = AgglomerativeClustering(n_clusters=num_k, affinity="euclidean", linkage="ward", compute_distances=True)
        model.fit(data_scaled)

        model_silhouette.append(silhouette_score(data_scaled, model.labels_))

    pyplot.plot(k_range, model_silhouette)
    pyplot.xticks(k_range)
    pyplot.xlabel("Number of clusters")
    pyplot.ylabel("Silhouette Coefficient")
    pyplot.title("Agglomerative Clustering (Silhouette method)")
    pyplot.show()

    mlsp.misc.print_title("> Agglomerative clustering with n_cluster=2", char="~")
    model = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="ward", compute_distances=True)

    data["Cluster Agglo."] = model.fit_predict(data_scaled)
    data_reduced = pandas.DataFrame(PCA(n_components=2).fit_transform(data[numeric_features]), columns=["pca1", "pca2"])

    pyplot.scatter(data_reduced["pca1"], data_reduced["pca2"], c=data["Cluster Agglo."], s=50)
    pyplot.title("Agglomerative clustering with n_cluster=2")
    pyplot.show()

    counts = numpy.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        curr_count = 0

        for child_i in merge:
            if child_i < n_samples:
                curr_count += 1
            else:
                curr_count += counts[child_i - n_samples]

        counts[i] = curr_count

    linkage_matrix = numpy.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix)
    pyplot.show()

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of USA arrests dataset in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
