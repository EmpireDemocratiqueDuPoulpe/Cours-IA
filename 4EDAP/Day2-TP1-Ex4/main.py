import pandas

# Remove value in column headers
data = pandas.read_csv("./data/billboard.csv")
df = pandas.melt(
    data,
    id_vars=["year", "artist", "track", "time", "date.entered"],
    value_vars=[
        "wk1", "wk2", "wk3", "wk4", "wk5", "wk6", "wk7", "wk8", "wk9", "wk10", "wk11", "wk12", "wk13", "wk14", "wk15",
        "wk16", "wk17", "wk18", "wk19", "wk20", "wk21", "wk22", "wk23", "wk24", "wk25", "wk26", "wk27", "wk28", "wk29",
        "wk30", "wk31", "wk32", "wk33", "wk34", "wk35", "wk36", "wk37", "wk38", "wk39", "wk40", "wk41", "wk42", "wk43",
        "wk44", "wk45", "wk46", "wk47", "wk48", "wk49", "wk50", "wk51", "wk52", "wk53", "wk54", "wk55", "wk56", "wk57",
        "wk58", "wk59", "wk60", "wk61", "wk62", "wk63", "wk64", "wk65", "wk66", "wk67", "wk68", "wk69", "wk70", "wk71",
        "wk72", "wk73", "wk74", "wk75", "wk76"
    ],
    var_name="week",
    value_name="rank"
)

# Reformat columns
df["date.entered"] = pandas.to_datetime(df["date.entered"], format="%Y-%m-%d")
df["week"] = df["week"].str.replace("wk", "").astype(int)
df["rank"] = df["rank"].astype("Int64")

print(df)
print(df.dtypes)
