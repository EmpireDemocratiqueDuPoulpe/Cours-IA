from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
import matplotlib.pyplot as pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/Tweets.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data)

    # Keep only interesting columns
    mlsp.misc.print_title("Keep only interesting columns")
    data = data[["text", "airline_sentiment"]]
    print(data.sample(n=5))

    fig, ax = pyplot.subplots(figsize=(10, 10))
    data.airline_sentiment.value_counts().plot(kind="pie", autopct="%1.1f%%")
    pyplot.title("Messages sentiment repartition")
    ax.set_xlabel("")
    ax.set_ylabel("")
    pyplot.show()

    # Text processing
    mlsp.misc.print_title("Text processing")

    def text_processing(text):
        text_cleaned = mlsp.text.clean(text)
        token = mlsp.text.tokenize(text_cleaned, language="english")
        token = mlsp.text.remove_stopwords(token, language="english")
        token = mlsp.text.remove_punctuation(token)
        token = mlsp.text.lemmatize(token)

        return token

    vectorized_data = mlsp.text.vectorization(data, col="text", analyzer=text_processing)
    tfidf_vectorized_data = mlsp.text.tfidf_vectorization(data, col="text", analyzer=text_processing)

    print(f"Vectorized:\n{Style.DIM}{Fore.WHITE}{vectorized_data.head()}")
    print(f"TFIDF Vectorized:\n{Style.DIM}{Fore.WHITE}{tfidf_vectorized_data.head()}")

    # Splitting dataset
    mlsp.misc.print_title("Splitting dataset")
    x_train, x_test, y_train, y_test = train_test_split(
        tfidf_vectorized_data,
        data["airline_sentiment"],
        test_size=0.20,
        random_state=100
    )
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}")

    # Get model
    mlsp.misc.print_title("Get model (LogisticRegression)")
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    print(f"Model accuracy: {Fore.LIGHTGREEN_EX}{round((model.score(x_test, y_test) * 100), 2)}%")

    # Model ratios
    ratios = pandas.DataFrame({"Feature": x_train.columns, "Ratio": model.coef_[0]})

    print((
        f"Important words for positive message detection:\n"
        f"{Style.DIM}{Fore.WHITE}{ratios.sort_values('Ratio', ascending=False).tail(n=7)}"
    ))
    print((
        f"Important words for negative message detection:\n"
        f"{Style.DIM}{Fore.WHITE}{ratios.sort_values('Ratio', ascending=False).head(n=7)}"
    ))

    # Generate word cloud
    mlsp.misc.print_title("Generate word cloud")

    def generate_wordcloud(word_list, title="Word cloud"):
        words = " ".join(word for word in word_list)
        cloud = WordCloud(
            background_color="white", colormap="Set1",
            width=1600, height=800,
            random_state=21,
            max_words=50,
            max_font_size=200
        ).generate(words)

        pyplot.figure(figsize=(12, 10))
        pyplot.title(title)
        pyplot.axis("off")
        pyplot.imshow(cloud, interpolation="bilinear")
        pyplot.show()

    generate_wordcloud(data["text"][data["airline_sentiment"] == "positive"], title="Positive words")
    generate_wordcloud(data["text"][data["airline_sentiment"] == "neutral"], title="Neutral words")
    generate_wordcloud(data["text"][data["airline_sentiment"] == "negative"], title="Negative words")

    # KNN Model
    mlsp.misc.print_title("Testing accuracy using KNN")
    model_knn = KNeighborsClassifier()
    model_knn.fit(x_train, y_train)

    print((
        f"Model score:\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ train → {round(model_knn.score(x_train, y_train) * 100, 2)}%{Fore.WHITE}{Style.DIM})\n"
        f"\t{Fore.WHITE}{Style.DIM}∟ test → {round(model_knn.score(x_test, y_test) * 100, 2)}%{Fore.WHITE}{Style.DIM})"
    ))

    predictions = model_knn.predict(x_test)
    predictions_probs = model_knn.predict_proba(x_test)[:, 1]
    predictions_comp = pandas.DataFrame({
        "Expected": y_test,
        "Predicted": predictions,
        "Probability": predictions_probs}
    )
    print(predictions_comp.sample(n=10))

    mlsp.metrics.confusion_matrix_from_estimator(model, x_test=x_test, y_test=y_test)

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of twitter messages in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
