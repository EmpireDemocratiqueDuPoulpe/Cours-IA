"""
Fatal encounter basic analysis
A. Lecomte <alexis.lecomte@supinfo.com>
"""

import os
import time
import re
import math
import numpy
import pandas
import geopandas
import scipy.stats
from geopandas import GeoDataFrame
import folium
from folium import plugins
import webbrowser
from tabulate import tabulate
import colorama
from colorama import Fore, Style
import matplotlib.pyplot as pyplot

colorama.init(autoreset=True)
pyplot.style.use("ggplot")

# Load CSV
data_fe = pandas.read_csv("./data/FatalEncounters.csv", dtype={
	"Unique ID": float,
	"Name": str,
	"Age": str,
	"Gender": "category",
	"Race": "category",
	"Race with imputations": "category",
	"URL of image (PLS NO HOTLINKS)": str,
	"Date of injury resulting in death (month/day/year)": "datetime64[ns]",
	"Location of injury (address)": str,
	"Location of death (city)": str,
	"State": str,
	"Location of death (zip code)": str,
	"Location of death (country)": str,
	"Full Address": str,
	"Latitude": str,
	"Longitude": float,
	"Agency or agencies involved": str,
	"Highest level of force": "category",
	"UID Temporary": str,
	"Name Temporary": str,
	"Armed/Unarmed": str,
	"Alleged weapon": str,
	"Aggressive physical movement": str,
	"Fleeing/Not fleeing": str,
	"Description Temp": str,
	"URL Temp": str,
	"Brief description": str,
	"Dispositions/Exclusions INTERNAL USE, NOT FOR ANALYSIS": object,
	"Intended use of force (Developing)": str,
	"Supporting document link": str,
	"Foreknowledge of mental illness? INTERNAL USE, NOT FOR ANALYSIS": object,
	"Unnamed: 32": float,
	"Unnamed: 33": float,
	"Unique ID formula": float,
	"Unique identifier (redundant)": float
})

# ### Fixes
# Drop "INTERNAL USE, NOT FOR ANALYSIS" columns
data_fe.drop(list(data_fe.filter(regex="INTERNAL USE, NOT FOR ANALYSIS")), axis=1, inplace=True)


# Fix - "Age" column
def _transform_age(age):
	if isinstance(age, str):
		if "-" in age and not age.startswith("-"):
			age_range = age.split("-")
			return round(float((int(age_range[0]) + int(age_range[1])) / 2), 0)
		else:
			return round(float(age), 0)

	return round(age, 0)


data_fe["Age"] = data_fe["Age"].map(lambda age: _transform_age(age))


# Fix - "Latitude" column
wrong_latitude_index = []


def _transform_latitude(lat):
	if isinstance(lat, str):
		try:
			return float(lat)
		except ValueError:
			wrong_latitude_index.append(data_fe.index[data_fe["Latitude"] == lat])
			return lat

	return lat


data_fe["Latitude"] = data_fe["Latitude"].map(lambda lat: _transform_latitude(lat))
data_fe.drop([index for sublist in wrong_latitude_index for index in sublist], inplace=True)
data_fe = data_fe.astype({"Latitude": float})

# Fix - "Race" column
race_uncertain = "Race unspecified"
data_fe["Race"].fillna(race_uncertain, inplace=True)
data_fe["Race"] = data_fe["Race"].map(lambda race: race.title())
data_fe.loc[(data_fe["Race"] == "Christopher Anthony Alexander"), "Race"] = race_uncertain
data_fe = pandas.concat([
	data_fe.drop("Race", axis=1),
	data_fe["Race"].str.get_dummies(sep="/").add_prefix("Race_")
], axis=1)

# Fix - "Armed/Unarmed" column
armed_uncertain = "Uncertain"


def _transform_is_armed(is_armed):
	if isinstance(is_armed, str):
		if (is_armed.lower() == "armed") or (is_armed.lower() == "unarmed"):
			return is_armed
		elif re.match(r"duplicate|none", is_armed, flags=re.IGNORECASE):
			return armed_uncertain
		else:
			return armed_uncertain
	else:
		return armed_uncertain


data_fe["Armed/Unarmed"] = data_fe["Armed/Unarmed"].map(lambda is_armed: _transform_is_armed(is_armed))
data_fe = data_fe.astype({"Armed/Unarmed": "category"})

# Fix - "Fleeing/Not fleeing" column
fleeing_uncertain = "Uncertain"


def _transform_is_fleeing(is_fleeing):
	if isinstance(is_fleeing, str):
		if re.match(r"duplicate|none", is_fleeing, flags=re.IGNORECASE):
			return fleeing_uncertain
		else:
			parts = is_fleeing.lower().split("/")

			if "fleeing" in parts:
				return "Fleeing"
			elif "not fleeing" in parts:
				return "Not fleeing"
			elif fleeing_uncertain.lower() in parts:
				return fleeing_uncertain
			else:
				return fleeing_uncertain
	else:
		return fleeing_uncertain


data_fe["Fleeing/Not fleeing"] = data_fe["Fleeing/Not fleeing"].map(lambda is_fleeing: _transform_is_fleeing(is_fleeing))
data_fe = data_fe.astype({"Fleeing/Not fleeing": "category"})

# #### First look
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ First look")
print(f"Shape: {Fore.LIGHTGREEN_EX}{data_fe.shape}")
print(data_fe.head(n=5))
print(f"Columns type: {Style.DIM}{Fore.WHITE}\n{data_fe.dtypes}")

# #### Column uniqueness
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Column uniqueness")


def is_column_unique(col_name, display_name=None):
	display_name = display_name if display_name else col_name
	is_unique = data_fe[col_name].is_unique
	print(f"Is the \"{display_name}\" unique?: {Fore.LIGHTGREEN_EX}{is_unique}")

	if not is_unique:
		columns_to_print = ["Unique ID"]

		if col_name not in columns_to_print:
			columns_to_print.append(col_name)

		print((
			f"The non-unique \"{display_name}\" are:\n"
			f"{Style.DIM}{Fore.WHITE}{data_fe[data_fe.duplicated(subset=[col_name], keep=False)][columns_to_print]}"
		))


is_column_unique("Unique ID", display_name="id")
is_column_unique("Name")

# #### Missing values per columns
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Missing values per columns")
print((
	"Total percent of missing values: "
	f"{Fore.LIGHTGREEN_EX}{round(((data_fe.isna().sum().sum() / (data_fe.shape[0] * data_fe.shape[1])) * 100), 2)}%"
))

percent_missing = round((data_fe.isna().sum() * 100) / len(data_fe), 2)
df_missing = pandas.DataFrame({"Column name": data_fe.columns, "Missing": percent_missing.values})
df_missing.sort_values("Missing", ascending=False, inplace=True)

print((
	f"Percent of missing values per columns:\n"
	f"{Style.DIM}{Fore.WHITE}{tabulate(df_missing, showindex=False, headers=df_missing.columns)}"
))

# Drop all columns with more than `drop_threshold`% of missing data
drop_threshold = 50
drop_columns = df_missing[df_missing["Missing"] >= drop_threshold]['Column name'].tolist()

if len(drop_columns) > 0:
	print((
		f"{Fore.YELLOW}Dropping all columns with more than {drop_threshold}% of missing data. Dropping "
		f"{drop_columns}..."
	))
	drop_prev_shape = data_fe.shape
	data_fe.drop(columns=drop_columns, axis=1, inplace=True)
	drop_new_shape = data_fe.shape
	drop_count = drop_prev_shape[1] - drop_new_shape[1]
	print(f"{Fore.YELLOW}Dropped {drop_count} column{'s' if drop_count > 1 else ''}. New shape: {drop_new_shape}")

# #### Duplicated values
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Duplicated values")
duplicated_data = data_fe.duplicated().sum()
print((
	f"Duplicated values: {Fore.LIGHTGREEN_EX}{duplicated_data}{Fore.RESET}"
	f" ({round((duplicated_data * 100) / data_fe.shape[0], 1)}%)"
))

# ### Basics analysis
# Frequency
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Basic analysis")
freq_columns = ["Gender", "State", "Armed/Unarmed", "Fleeing/Not fleeing"]
freq_n_cols = 2
freq_n_rows = math.ceil(len(freq_columns) / freq_n_cols)
freq_fig = pyplot.figure(figsize=(4 * freq_n_cols, 3 * freq_n_rows))
freq_fig.suptitle("Frequencies of qualitative data")

for index, col in enumerate(freq_columns):
	ax = pyplot.subplot(freq_n_rows, freq_n_cols, index + 1)

	data_count = data_fe[[col]].value_counts()
	labels = [label[0] for label in data_count.index.tolist()]
	bars = data_count.plot(ax=ax, kind="bar", legend=True, label=col.title())
	ax.set_xlabel("")
	ax.set_xticks(ticks=numpy.arange(0, len(data_count)), labels=labels, rotation=45, fontsize=10)

	for bars in ax.containers:
		ax.bar_label(bars)

pyplot.tight_layout()
pyplot.show()

# Mean - Median - Mode - STD - Skewness - Kurtosis
numerical_columns = ["Age"]

for column in numerical_columns:
	mmmssk_fig, (mmmssk_box_ax, mmmssk_hist_ax) = pyplot.subplots(
		nrows=2, ncols=1,
		sharex=True,
		gridspec_kw={"height_ratios": (0.2, 1)}
	)
	mmmssk_fig.suptitle(f"Statistics of \"{column}\" column")

	mean = data_fe[column].mean()
	median = data_fe[column].median()
	mode = data_fe[column].mode().values[0]

	data_fe[column].plot(ax=mmmssk_box_ax, kind="box", vert=False, legend=True)
	mmmssk_box_ax.axvline(mean, color="darkred", linestyle="--")
	mmmssk_box_ax.axvline(median, color="darkgreen", linestyle="-")
	mmmssk_box_ax.axvline(mode, color="darkblue", linestyle="-")

	data_fe[column].plot(ax=mmmssk_hist_ax, kind="hist", legend=True)
	mmmssk_hist_ax.axvline(mean, color="darkred", linestyle="--", label="Mean")
	mmmssk_hist_ax.axvline(median, color="darkgreen", linestyle="-", label="Median")
	mmmssk_hist_ax.axvline(mode, color="darkblue", linestyle="-", label="Mode")

	mmmssk_box_ax.set_yticklabels([""])
	mmmssk_hist_ax.legend()

	standard_deviation = data_fe[column].std()
	skewness = data_fe[column].skew()
	kurtosis = data_fe[column].kurtosis()

	pyplot.figtext(0.73, 0.58, f"Other stats:", fontsize=8, fontweight="bold")
	pyplot.figtext(0.73, 0.54, f"Standard deviation: {round(standard_deviation, 3)}", fontsize=8)
	pyplot.figtext(0.73, 0.5, f"Skewness: {round(skewness, 3)}", fontsize=8)
	pyplot.figtext(0.73, 0.46, f"Kurtosis: {round(kurtosis, 3)}", fontsize=8)

	pyplot.tight_layout()
	pyplot.subplots_adjust(right=0.70)
	pyplot.show()

# #### Repartition of fatal encounters
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Repartition of fatal encounters")
repart_fig = pyplot.figure()
repart_size = (2, 2)
repart_grid = repart_fig.add_gridspec(repart_size[0], repart_size[1])

repart_fig.suptitle("Repartition of fatal encounters")

# Fatal encounters per gender per age
male_labels = data_fe["Age"][data_fe["Gender"] == "Male"].drop_duplicates(keep="first", inplace=False).dropna().sort_values()
male_values = data_fe["Age"][data_fe["Gender"] == "Male"].value_counts().sort_index()

female_labels = data_fe["Age"][data_fe["Gender"] == "Female"].drop_duplicates(keep="first", inplace=False).dropna().sort_values()
female_values = data_fe["Age"][data_fe["Gender"] == "Female"].value_counts().sort_index()

gender_ax = pyplot.subplot(repart_grid[0, :-1])
gender_ax.plot(male_labels, male_values.tolist(), label="Male")
gender_ax.plot(female_labels, female_values.tolist(), label="Female")

gender_ax.set_xlabel("Age")
gender_ax.legend()

# Fatal encounters per ethnic group
# Revert one hot encoding
ethnic_data = data_fe[[column for column in data_fe if column.startswith("Race_")]].idxmax(1)
ethnic_data = ethnic_data.map(lambda ethnic: ethnic.replace("Race_", "").title())

ethnic_values = ethnic_data.value_counts().sort_values(ascending=False)
ethnic_labels = pandas.Series(data=ethnic_values.index, index=ethnic_values)

ethnic_ax = pyplot.subplot(repart_grid[1, :-1])
ethnic_ax.bar(ethnic_labels, ethnic_values.tolist())

ethnic_ax.set_xticks(
	ticks=numpy.arange(0, len(ethnic_labels)),
	labels=ethnic_labels,
	rotation=45, fontsize=8, ha="right"
)


# Fatal encounters per state
def autopct_func(pct):
	return f"{round(pct, 2)}%" if pct > 5 else ""


# Get states
states = data_fe["State"].sort_values()
states_values = states.value_counts().sort_values(ascending=False)

# Merge states with low values
states_threshold = 1
states_to_merge = states_values[states_values <= ((states_values.sum() * states_threshold) / 100)]
states_values = pandas.concat([
	states_values.drop(states_values.tail(n=len(states_to_merge)).index),
	pandas.Series(data=sum(states_to_merge), index=["Rest"])
])

# Get states labels
states_labels = pandas.Series(data=states_values.index, index=states_values)

# Plot
states_ax = pyplot.subplot(repart_grid[0:, -1])
states_ax.pie(states_values.tolist(), labels=states_labels, autopct=autopct_func, textprops={"fontsize": 8})
states_ax.axis("equal")
states_ax.set_title("Repartition per states")

# Show plot
pyplot.tight_layout()
pyplot.show()


# Fatal encounters per locations
class Map:
	def __init__(self, crs="EPSG:3857", **kwargs):
		self.crs = crs
		self.lat_lon_df = None
		self.geometry_df = None

		self.map = folium.Map(tiles=None, crs=self.crs.replace(":", ""), **kwargs)
		self._add_tile_layers()

	def _add_tile_layers(self):
		layers = [
			{"name": "openstreetmap", "display_name": "Open Street Map"},
			{"name": "stamentoner", "display_name": "Stamen toner"},
			{"name": "cartodbpositron", "display_name": "CartoDB (Light)"},
			{"name": "cartodbdark_matter", "display_name": "CartoDB (Dark)"},
		]

		for layer in layers:
			folium.TileLayer(layer["name"], name=layer["display_name"]).add_to(self.map)

	def load_from_gcs(self, df, lat_name="Latitude", lon_name="Longitude"):
		self.lat_lon_df = df[[lat_name, lon_name]].copy()
		self.geometry_df = GeoDataFrame(
			df,
			geometry=geopandas.points_from_xy(self.lat_lon_df[lon_name], self.lat_lon_df[lat_name]),
			crs=self.crs
		)

	def add_heatmap(self, color_map=False, layered=False, **kwargs):
		heat_data = [[point.xy[1][0], point.xy[0][0]] for point in self.geometry_df.geometry if not point.is_empty]

		if color_map:
			heat_map = folium.plugins.HeatMap(heat_data, gradient={0.4: "blue", 0.65: "lime", 1: "red"}, **kwargs)
		else:
			heat_map = folium.plugins.HeatMap(heat_data, **kwargs)

		if layered:
			feature_group = folium.FeatureGroup(name="Heat Map")
			feature_group.tile_name = "Heat Map"
			heat_map.add_to(feature_group)

		heat_map.add_to(self.map)

		if layered:
			folium.LayerControl().add_to(self.map)

	def fit_bounds(self):
		south_west_bound = self.lat_lon_df.min().values.tolist()
		north_east_bound = self.lat_lon_df.max().values.tolist()
		self.map.fit_bounds([south_west_bound, north_east_bound])

	def open(self, output_dir="./temp", filename=None):
		path = os.path.join(output_dir, (filename if filename else f"map-{time.time()}.html"))

		self.map.save(path)
		webbrowser.open(path)


encounter_map = Map()
encounter_map.load_from_gcs(data_fe, lat_name="Latitude", lon_name="Longitude")
encounter_map.add_heatmap(layered=True, min_opacity=0.4, radius=20, blur=20)
encounter_map.fit_bounds()
encounter_map.open()

# #### Chi square test
print(f"{Style.BRIGHT}{Fore.GREEN}\\\\ Chi square test")
# Contingency table
contingency = pandas.crosstab(
	index=data_fe["Fleeing/Not fleeing"],
	columns=data_fe["Armed/Unarmed"],
	normalize="index"
).round(4)
contingency_numpy = contingency.to_numpy()

chi_labels_armed = contingency.columns.tolist()
chi_labels_fleeing = contingency.index.tolist()

# Plot to heatmap
chi_fig, chi_ax = pyplot.subplots()

chi_ax.imshow(contingency_numpy)
chi_ax.set_xticks(numpy.arange(len(chi_labels_armed)), labels=chi_labels_armed)
chi_ax.set_yticks(numpy.arange(len(chi_labels_fleeing)), labels=chi_labels_fleeing)

for i in range(len(chi_labels_fleeing)):
	for j in range(len(chi_labels_armed)):
		chi_ax.text(j, i, contingency_numpy[i, j], ha="center", va="center", color="w")

# Interpret statistics
stat, pvalue, dof, expected = scipy.stats.chi2_contingency(contingency)

print(f"Degree of freedom: {Fore.LIGHTGREEN_EX}{dof}")
print(f"Expected frequencies: {Fore.LIGHTGREEN_EX}{expected}")

prob = 0.95
critical = scipy.stats.chi2.ppf(prob, dof)
print((
	f"Interpret the test-statistic: "
	f"{Style.DIM}{Fore.WHITE}(Probability: {round(prob, 3)} | Critical: {round(critical, 3)} | Stat: {round(stat, 3)}) "
	f"{Style.RESET_ALL}{Fore.LIGHTGREEN_EX}>>> Dependent: {abs(stat) >= critical}"
))

alpha = 1.0 - prob
print((
	f"Interpret the p-value: "
	f"{Style.DIM}{Fore.WHITE}(Significance: {round(alpha, 3)} | P-Value: {round(pvalue, 3)}) "
	f"{Style.RESET_ALL}{Fore.LIGHTGREEN_EX}>>> Dependent: {pvalue <= alpha}"
))

# Finish plot
chi_ax.set_title(
	"Chi square test: Independence of \"Armed/Unarmed\" and \"Fleeing/Not fleeing\"",
	loc="center",
	wrap=True
)
chi_ax.grid(False)
chi_fig.tight_layout()
chi_fig.subplots_adjust(top=0.85)
pyplot.show()
