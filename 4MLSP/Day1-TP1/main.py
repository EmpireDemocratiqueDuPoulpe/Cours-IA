import colorama
from colorama import Fore, Back, Style
import pandas
import numpy
import scipy.stats as scistats
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

colorama.init(autoreset=True)

# Read CSV
data = pandas.read_csv("./data/HeartDiseaseUCI.csv")

# First look
print(f"{Fore.GREEN}#### First look ##################################################################################")
print(f"Shape: {Fore.LIGHTGREEN_EX}{data.shape}")
print(data.head())
print(data.dtypes)

print(f"{Fore.GREEN}#### Missing values ##############################################################################")
print(f"Missing:\n{Fore.LIGHTGREEN_EX}{data.isna().sum()}")

print(f"{Fore.GREEN}#### Replace \"num\" column ######################################################################")
data = data.assign(num=lambda n: (n["num"] > 0).astype(int))
data = data.rename({"num": "disease"}, axis="columns")
print(data.head())

print(f"{Fore.GREEN}#### Study #######################################################################################")
conf_interval = scistats.t.interval(alpha=0.95, df=(len(data) - 1), loc=numpy.mean(data), scale=scistats.sem(data))
print(f"Confidence interval: {Fore.LIGHTGREEN_EX}{conf_interval}")

print(f"{Fore.GREEN}#### Splitting dataset ###########################################################################")
data_train = data.sample(frac=0.8, random_state=200)
data_test = data.drop(data_train.index)
print(f"Train data: {Fore.LIGHTGREEN_EX}{data_train.shape}")
print(f"Test data: {Fore.LIGHTGREEN_EX}{data_test.shape}")

print(f"{Fore.GREEN}#### Transform numeric and categorical values ####################################################")
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant")),
    ("encoder", OrdinalEncoder())
])

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "disease"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

data_train = preprocessor.fit_transform(data_train)
data_test = preprocessor.fit_transform(data_test)
print(f"Train data: {Fore.LIGHTGREEN_EX}{data_train}")
print(f"Test data: {Fore.LIGHTGREEN_EX}{data_test}")
