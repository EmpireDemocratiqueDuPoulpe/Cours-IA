import re
import string
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean(text):
    text = text.lower()  # Lower case
    text = re.sub(r"\W/_", " ", text)  # Remove special characters
    text = re.sub(r"\d", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Replace multi-space by single one
    text = re.sub(r"^\s/\s$", "", text)  # Trim space at beginning and end
    text = re.sub(r"\s+[a-zA-Z]\s+", "", text)  # Remove single characters
    text = re.sub(r"^@\S+|\s@\S+", "<mention>", text)  # Replace mention
    text = re.sub(r"^#\S+|\s#\S+", "<hashtag>", text)  # Replace hashtag
    text = re.sub(r"https?:\S+/http?:\S", "<link>", text)  # Replace links
    text = _save_emoticon(text)  # Replace emoticons

    return text


def _save_emoticon(text):
    text = re.sub(r";p|;P|:p|:P|xp|xP|=p|=P|:‑P|X‑P|x‑p|:‑p|:‑Þ|:‑þ|:‑b|>:P|d:|:b|:þ|:Þ", "<emoticon_tongue>", text)
    text = re.sub(
        r":‑\)|:\)|:-]|:]|:->|:>|8-\)|8\)|:-}|:}|:o\)|:c\)|:\^\)|=]|=\)|:-\)\)|:'‑\)|:'\)|:\"D'",
        "<emoticon_happy>",
        text
    )
    text = re.sub(r":‑D|:D|8‑D|8D|=D|=3|B\^D|c:|C:|x‑D|xD|X‑D|XD", "<emoticon_laugh>", text)
    text = re.sub(r":‑\(|:‑c|:c|:‑<|:<|:‑\[|:\[|>:\[|:{|:@|:\(|;\(|:'‑\(|:'\(|:=\(|v.v", "<emoticon_sad>", text)
    text = re.sub(r"D‑':|D:<|D:|D8|D;|D=|DX", "<emoticon_disgust>", text)
    text = re.sub(r":‑O|:O|:‑o|:o|:-0|8‑0|>:O|=O|=o|=0|O_O|o_o|O-O|o‑o|O_o|o_O", "<emoticon_surprise>", text)
    text = re.sub(r":-3|:3|=3|x3|X3|>:3", "<emoticon_cat>", text)
    text = re.sub(r":-\*|:\*|:×|<3", "<emoticon_love>", text)
    text = re.sub(r";‑\)|;\)|\*-\)|\*\)|;‑]|;]|;\^\)|;>|:‑,|;D|;3|:‑J", "<emoticon_wink>", text)
    text = re.sub(r":-/ |>.<|>_<|:/|:‑.|>:\|>:/|:\|=/|=\|:L|=L|:S", "<emoticon_skeptical>", text)
    text = re.sub(r"<_<|>_>|<.<|>.>|:$|://|://3|:‑X|:X|:‑#|:#|:‑&|:&|%‑\)|%\)", "<emoticon_embarrassed>", text)
    text = re.sub(r"8-X|8=X|x-3|x=3|X_X|x_x", "<emoticon_dead>", text)

    return text


def tokenize(text, language="english", preserve_line=False):
    return word_tokenize(text, language=language, preserve_line=preserve_line)


def stem(token):
    return [PorterStemmer().stem(word) for word in token]


def lemmatize(token):
    return [WordNetLemmatizer().lemmatize(word) for word in token]


def remove_stopwords(token, language="english"):
    s_words = stopwords.words(language)
    return [word for word in token if token not in s_words]


def remove_punctuation(token):
    return [word for word in token if word not in string.punctuation]


def vectorization(data, col, analyzer, max_features=None):
    vectorized = CountVectorizer(analyzer=analyzer, max_features=max_features)
    x = vectorized.fit_transform(data[col])

    data_text = pandas.DataFrame(x.toarray())
    data_text.columns = vectorized.get_feature_names_out()

    return data_text


def tfidf_vectorization(data, col, analyzer, max_features=None):
    tfidf_vectorized = TfidfVectorizer(analyzer=analyzer, max_features=max_features)
    x = tfidf_vectorized.fit_transform(data[col])

    data_text = pandas.DataFrame(x.toarray())
    data_text.columns = tfidf_vectorized.get_feature_names_out()

    return data_text
