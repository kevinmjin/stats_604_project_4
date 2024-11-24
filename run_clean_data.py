import pandas as pd
import numpy as np
import os, pickle, janitor, requests
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

raw_data_url = "https://raw.githubusercontent.com/kevinmjin/winter_weather_data/refs/heads/main/winter_weather_data/weather_final.csv"

ogdf = getDataFrame(f'{raw_data_url}')
mydf = ogdf.copy()

missing_min = mydf[mydf['temp_min'] == 0]  # Example subset, you can define your own subset
missing_min_location = missing_min['location'].tolist()
missing_min_date = missing_min['date'].tolist()
missing_min_rows = missing_min.index.tolist()

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
               32 # KBOI 2021-12-02
               ] 

# Refill unreasonable values
for i in range(len(missing_min_rows)):
    mydf.loc[missing_min_rows[i], 'temp_min'] = min_refills[i]
    print(missing_min_location[i], missing_min_date[i],'refilled with temp_min= ', min_refills[i])
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
