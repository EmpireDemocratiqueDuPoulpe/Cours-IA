import pandas

# Remove value in column headers
data = pandas.read_csv("./data/tuberculosis.csv")
tidyDataset = pandas.melt(
    data,
    id_vars=["country", "year"],
    value_vars=[
        "m014", "m1524", "m2534", "m3544", "m4554", "m5564", "m65", "mu", "f014", "f1524", "f2534", "f3544", "f4554",
        "f5564", "f65", "fu"
    ],
    var_name="sex/age",
    value_name="tuberculosis cases"
)

# Split sex and age
tidyDataset[["sex", "age"]] = tidyDataset["sex/age"].str.extract("(?P<sex>^[a-z]{1})(?P<age>.*$)", expand=True)
tidyDataset.drop(columns=["sex/age"], inplace=True)
tidyDataset = tidyDataset.reindex(columns=["country", "year", "sex", "age", "tuberculosis cases"])

print(tidyDataset)
