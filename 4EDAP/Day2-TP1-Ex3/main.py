import pandas
import numpy

# Remove value in column headers
data = pandas.read_csv("./data/weather.csv")
df = pandas.melt(
    data,
    id_vars=["id", "year", "month", "element"],
    value_vars=[
        "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"
    ]
)

# Pivot table
df = pandas.pivot_table(df, index=["id", "year", "month"], columns=["element"], values="value", aggfunc=numpy.sum)

print(df)
