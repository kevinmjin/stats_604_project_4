# %%
import pandas as pd
import numpy as np
import os, pickle, janitor, requests
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# %%
## Get Location Code from the Data Frame
def getLocationCode(text):
    return text.split("/")[-1]

## Apply it to a Pandas Series
def getLocationCodeSeries(pd_series):
    return pd_series.apply(getLocationCode)

## Get the Data Frame
def getDataFrame(url: str):
    df = (
        pd.read_csv(url)
        .pipe(janitor.clean_names, strip_underscores=True)
        .assign(date=lambda x: pd.to_datetime(x.date),
                location=lambda x: getLocationCodeSeries(x.location))
        .drop(columns=["unnamed_0", "actual_time", "dew_point"])
        .rename(columns={"high_temp": "temp_max",
                         "low_temp": "temp_min",
                         "day_average_temp": "temp_mean",
                         "high": "dew_point_max",
                         "low": "dew_point_min",
                         "average": "dew_point_mean",
                         "visibiilty": "visibility",})
    )
    return df

# %%
raw_data_url = "https://raw.githubusercontent.com/kevinmjin/winter_weather_data/refs/heads/main/winter_weather_data/weather_final.csv"

ogdf = getDataFrame(f'{raw_data_url}')
mydf = ogdf.copy()

# %% [markdown]
# ## Datacleaning

# %%
mydf.describe()

# %%
mydf[mydf['temp_max'] == 201]

# %%
mydf.loc[3317, 'temp_max'] = 50

# %%
mydf[mydf['temp_min'] == 0]

# %%
missing_min = mydf[mydf['temp_min'] == 0]  # Example subset, you can define your own subset
missing_min_location = missing_min['location'].tolist()
missing_min_date = missing_min['date'].tolist()
missing_min_rows = missing_min.index.tolist()

# %%
min_refills = [36, # KSFO 2020-12-01
               43, # KSEA 2022-11-25
               38, # KDCA 2021-11-09
               45, # KPWM 2019-10-27
               73, # KMIA 2020-11-13
               72, # KMIA 2023-11-18
               73, # KMIA 2023-11-25
               75, # KMIA 2024-11-11
               36, # KSWF 2019-11-27
               39, # KSWF 2019-12-14
               30, # KSWF 2020-11-17
               41, # KSWF 2020-11-21
               46, # KSWF 2020-11-27
               37, # KSWF 2020-11-28
               30, # KSWF 2020-12-03
               43, # KSWF 2021-12-16
               39, # KSWF 2022-11-03
               32, # KSWF 2024-11-16
               0, # PANC 2020-12-05
               0, # PANC 2021-11-16
               0,# PANC 2022-11-28
               0, # PANC 2022-11-07
               33, # KBOI 2019-11-13
               32, # KBOI 2021-12-02
               31, # KJFK 2019-12-02
               52, # KJFK 2020-10-28
               ] 

# %%
# Refill unreasonable values
for i in range(len(missing_min_rows)):
    mydf.loc[missing_min_rows[i], 'temp_min'] = min_refills[i]
    print(missing_min_location[i], missing_min_date[i],'refilled with temp_min= ', min_refills[i])


# %%
locations_dict = {
                "Anchorage": "PANC",
                "Boise": "KBOI",
                "Chicago": "KORD",
                "Denver": "KCOS",
                "Detroit": "KDTW",
                "Honolulu": "PHNL",
                "Houston": "KIAH",
                "Miami": "KMIA",
                "Minneapolis": "KMSP",
                "Oklahoma City": "KOKC",
                "Nashville": "KBNA",
                "New York": "KJFK",
                "Phoenix": "KPHX",
                "Portland ME": "KPWM",
                "Portland OR": "KPDX",
                "Salt Lake City": "KSLC",
                "San Diego": "KSAN",
                "San Francisco": "KSFO",
                "Seattle": "KSEA",
                "Washington DC": "KDCA"
            }
## keep only the locations of locations_dict_values
mydf = mydf[mydf['location'].isin(locations_dict.values())]
mydf.to_csv('data/data_cleaned.csv')

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Time Series Analysis

# %%
import matplotlib.pyplot as plt
mydf['date'] = pd.to_datetime(mydf['date'])
mydf['year']= mydf['date'].dt.year
cities = mydf['location'].unique()

plt.figure(figsize=(15, 8))
for city in cities:
    city_data = mydf[mydf['location'] == city][mydf['year'] == 2023]
    plt.plot(city_data['date'], city_data['temp_mean'], label=f'{city} Mean Temp', alpha=0.6)

plt.title("Daily Mean Temperature Trends Across Cities", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°F)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid()
plt.show()


# %% [markdown]
# ### Correlations

# %%
import seaborn as sns

numerical_columns = ['temp_max', 'temp_min', 'temp_mean', 'precipitation', 
                     'dew_point_max', 'dew_point_min', 'dew_point_mean', 
                     'max_wind_speed', 'visibility', 'sea_level_pressure']
correlation_data = mydf[numerical_columns]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Weather Variables", fontsize=16)
plt.show()

# Scatter plots for key correlations (Example)
# Plot temp_mean vs. dew_point_mean
plt.figure(figsize=(8, 6))
plt.scatter(mydf['dew_point_mean'], mydf['temp_mean'], alpha=0.6)
plt.title("Scatter Plot: Temp Mean vs. Dew Point Mean", fontsize=16)
plt.xlabel("Dew Point Mean", fontsize=12)
plt.ylabel("Temperature Mean (°F)", fontsize=12)
plt.grid()
plt.show()

# Plot temp_mean vs. sea_level_pressure
plt.figure(figsize=(8, 6))
plt.scatter(mydf['sea_level_pressure'], mydf['temp_mean'], alpha=0.6)
plt.title("Scatter Plot: Temp Mean vs. Sea Level Pressure", fontsize=16)
plt.xlabel("Sea Level Pressure", fontsize=12)
plt.ylabel("Temperature Mean (°F)", fontsize=12)
plt.grid()
plt.show()


# %% [markdown]
# ### City-wise Analysis
# 

# %%

city_summary = mydf.groupby('location').agg({
    'temp_min': [ 'median', 'std'],
    'temp_mean': [ 'median', 'std'],
    'temp_max': ['median', 'std'],
    'precipitation': 'mean',
    'max_wind_speed': 'mean'
}).reset_index()


city_summary.columns = ['_'.join(col).strip('_') for col in city_summary.columns]

print("City-wise Weather Summary:")
print(city_summary)

# Boxplot for temperature distributions across cities
plt.figure(figsize=(15, 8))
mydf.boxplot(column='temp_mean', by='location', grid=False, rot=90, figsize=(15, 8))
plt.title("Temperature Mean Distribution Across Cities", fontsize=16)
plt.xlabel("City", fontsize=12)
plt.ylabel("Temperature Mean (°F)", fontsize=12)
plt.suptitle("")  # Remove automatic title
plt.show()

# Bar chart: Average precipitation across cities
city_avg_precipitation = mydf.groupby('location')['precipitation'].mean().reset_index()

plt.figure(figsize=(15, 8))
plt.bar(city_avg_precipitation['location'], city_avg_precipitation['precipitation'], alpha=0.7)
plt.title("Average Precipitation by City", fontsize=16)
plt.xlabel("City", fontsize=12)
plt.ylabel("Average Precipitation (inches)", fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




# %%
import folium
import pandas as pd

airport_data = {
    'ICAO': ['KSAN', 'KSFO', 'KSEA', 'KDCA', 'KPHX', 'KPWM', 'KPDX', 'KSLC',
             'KMIA', 'KMSP', 'KOKC', 'KBNA', 'KSWF', 'PANC', 'KMDW', 'PHNL',
             'KIAH', 'KDTW', 'KBOI', 'KORD', 'KCOS', 'KJFK'],
    'Name': ['San Diego Intl', 'San Francisco Intl', 'Seattle-Tacoma Intl',
             'Reagan National', 'Phoenix Sky Harbor', 'Portland Intl Jetport',
             'Portland Intl', 'Salt Lake City Intl', 'Miami Intl', 'Minneapolis-St. Paul Intl',
             'Will Rogers World', 'Nashville Intl', 'Stewart Intl', 'Ted Stevens Anchorage Intl',
             'Chicago Midway', 'Honolulu Intl', 'George Bush Intercontinental',
             'Detroit Metropolitan', 'Boise Airport', 'Chicago O\'Hare', 
             'Colorado Springs', 'John F. Kennedy Intl'],
    'Latitude': [32.7338, 37.6188, 47.4502, 38.8512, 33.4353, 43.646, 45.5898,
                 40.7861, 25.7933, 44.8848, 35.3931, 36.1263, 41.5041, 61.1743, 
                 41.785, 21.3187, 29.9902, 42.2162, 43.5644, 41.9742, 38.8058, 40.6413],
    'Longitude': [-117.1933, -122.375, -122.3088, -77.0377, -112.0078, -70.3081, 
                  -122.5951, -111.9776, -80.2906, -93.2223, -97.6016, -86.6774,
                  -74.1048, -149.998, -87.7524, -157.9224, -95.3368, -83.3554, 
                  -116.223, -87.9073, -104.7007, -73.7781]
}

# Convert to DataFrame
df = pd.DataFrame(airport_data)

# Base map
usa_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Add airport markers
for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['ICAO']} - {row['Name']}",
        tooltip=f"{row['ICAO']}",
    ).add_to(usa_map)

usa_map.save("airport_map.html")



