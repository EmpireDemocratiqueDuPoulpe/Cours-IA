import pandas

# Remove value in column headers
data = pandas.read_csv("./data/pew.csv")
tidyDataset = pandas.melt(
    data,
    id_vars=["religion"],
    value_vars=[
        "<$10k", "$10-20k", "$20-30k", "$30-40k", "$40-50k", "$50-75k", "$75-100k", "$100-150k", "Don't know/refused"
    ],
    var_name="income",
    value_name="believers per income"
)

print(tidyDataset)
