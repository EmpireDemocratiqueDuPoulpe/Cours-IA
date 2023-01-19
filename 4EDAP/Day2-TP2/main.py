import pandas


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
data = pandas.read_csv("./data/datasetSW.csv")
df = data

# 2. First look at the data
print(f"{TStyles.HEADER}{TStyles.BOLD}#### First look ##################################################{TStyles.ENDC}")
print(f"{TStyles.GREEN}## First five rows{TStyles.ENDC}")
print(df.head(n=5))
print(f"{TStyles.GREEN}## Data shape{TStyles.ENDC}")
print(df.shape)
print(f"{TStyles.GREEN}## Columns types{TStyles.ENDC}")
print(df.dtypes)

# 3. Is the name unique ?
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Uniqueness of the name column ###############################{TStyles.ENDC}")

isNameUnique = df["name"].is_unique
print(f"{TStyles.UNDERLINE}Is the name unique?:{TStyles.ENDC} {isNameUnique}")

if not isNameUnique:
    print(f"{TStyles.UNDERLINE}The non-unique names are:{TStyles.ENDC}")
    print(df[df.duplicated(subset=["name"], keep=False)])

    print(f"{TStyles.WARNING}Removing duplicates...{TStyles.ENDC}")
    df.drop_duplicates(subset=["name"], keep="first", inplace=True)

# 4. isnull & notnull
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Null values #################################################{TStyles.ENDC}")
print(f"{TStyles.GREEN}## \"True\" when it's null {TStyles.ENDC}")
print(pandas.isnull(df))

print(f"{TStyles.GREEN}## \"True\" when it's not null {TStyles.ENDC}")
print(pandas.notnull(df))

# 5. Missing height
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Characters without height ###################################{TStyles.ENDC}")
print(df[df["height"].isna()])

# 6. How many empty values ?
print(f"{TStyles.HEADER}{TStyles.BOLD}#### How many missing values? ####################################{TStyles.ENDC}")
print(df.isna().sum())

# 7. Drop every row with 8 or more missing values out of 10
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Drop every row with 8 or more missing values out of 10 ######{TStyles.ENDC}")

maxMissingValue = 7
df.dropna(thresh=df.shape[1] - maxMissingValue, axis=0, inplace=True)
print(df)

# 8. Drop the column with the most of missing values
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Drop the column with the most of missing values #############{TStyles.ENDC}")

targetedColumn = df.isnull().sum().idxmax()
print(f"{TStyles.WARNING}Removing the \"{targetedColumn}\" column...{TStyles.ENDC}")
del df[targetedColumn]

# 11. Fill height column NA with the column median
print(f"{TStyles.HEADER}{TStyles.BOLD}#### Fill height column NA with the column median ################{TStyles.ENDC}")
df.fillna(value=df["height"].median(), inplace=True)
print(df)

# 12.
print(df[df.duplicated()])
