import pandas

# Load dataset
data2014 = pandas.read_csv("./data/babyNames2014.csv")
data2015 = pandas.read_csv("./data/babyNames2015.csv")

# Add study year to each dataframes
data2014["study_year"] = 2014
data2015["study_year"] = 2015

# Concat dataframes
df = pandas.concat([data2014, data2015])

# Reorder the output
df = df.reindex(columns=["study_year", "rank", "name", "sex", "frequency"])
df = df.sort_values(["study_year", "rank"], ascending=[True, True])

print(df)
