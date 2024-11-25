import pandas as pd
import numpy as np
import os, pickle, janitor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


## Impor the Cleaned Up Data Frame
def getCleanDataFrame(folder_path: str) -> pd.DataFrame:
    files = os.listdir(folder_path)
    ## keep only the csv files
    files = [file for file in files if file.endswith(".csv")]
    df = pd.concat([pd.read_csv(folder_path + file)
                    .pipe(janitor.clean_names, strip_underscores=True)
                    .drop(columns=["unnamed_0"])
                    .assign(date=lambda x: pd.to_datetime(x.date),
                            temp_max=lambda x:pd.to_numeric(x.temp_max),
                            temp_min=lambda x:pd.to_numeric(x.temp_min),
                            temp_mean=lambda x:pd.to_numeric(x.temp_mean),
                            precipitation=lambda x:pd.to_numeric(x.precipitation),
                            dew_point_max=lambda x:pd.to_numeric(x.dew_point_max),
                            dew_point_min=lambda x:pd.to_numeric(x.dew_point_min),
                            dew_point_mean=lambda x:pd.to_numeric(x.dew_point_mean),
                            max_wind_speed=lambda x:pd.to_numeric(x.max_wind_speed),
                            visibility=lambda x:pd.to_numeric(x.visibility),
                            sea_level_pressure=lambda x:pd.to_numeric(x.sea_level_pressure)
                    )
                     for file in files])
    return df

######## Created Lagged Data
### df: DataFrame
### lag: int
### to_lag_columns: list
def allLaggedDataAvail(df: pd.DataFrame, lag: int, to_lag_columns: list) -> pd.DataFrame:
    df = df.sort_values(by=["date"])
    df['day_diff'] = df['date'].diff().dt.days
    df['has_all_lagged'] = (
        df['day_diff']
        .rolling(window=lag)
        .apply(lambda x: np.all(x == 1), raw=True)
    )


    for col in to_lag_columns:
        for i_lag in range(1, lag + 1):
            df[f'{col}_lag_{i_lag}'] = df[f'{col}'].shift(i_lag)
    df = df[df['has_all_lagged'] == 1].copy()
    df.dropna(subset=[f'{col}_lag_{lag}' for lag in range(1, lag + 1)], inplace=True)

    # Drop temporary columns
    df = df.drop(columns=['day_diff', 'has_all_lagged'])
    #df = df.drop(columns=to_lag_columns[3:])
    return df.sort_values(by = "date").reset_index(drop=True)

## Create a Model for each Location
class LocationModel():
    def __init__(self, location: str, df: pd.DataFrame, lag: int):
        self.to_lag_columns = ['temp_max',
                               'temp_min',
                               'temp_mean',
                               'precipitation',
                               'dew_point_max',
                               'dew_point_min',
                               'dew_point_mean',
                               'max_wind_speed',
                               'visibility',
                               'sea_level_pressure']
        self.lag = lag
        
        self.location = location
        self.df = df.copy()
        self.lagged_df = allLaggedDataAvail(df, lag, self.to_lag_columns)

    def fit(self):
        X_ = (
            self.lagged_df
            .drop(columns=self.to_lag_columns)
            .drop(columns = ["date", "location"])
            .to_numpy()
        )
        y_ = self.lagged_df[self.to_lag_columns].to_numpy()
        
        self.scaler = StandardScaler()
        X_ = self.scaler.fit_transform(X_)
        
        ## Define the Kernel
        gp = GPR(normalize_y=True, n_restarts_optimizer=10)

        ## Define the Parameters
        param_grid = {"alpha": np.logspace(1e-12,10,100),
                      "kernel": [RBF(l, length_scale_bounds="fixed") for l in np.logspace(-5,5,100)]}
        ## Define the Grid Search
        tscv = TimeSeriesSplit(n_splits=5)
        self.model = GridSearchCV(gp, param_grid=param_grid, cv=tscv, n_jobs=-1, 
                                  scoring = "neg_mean_squared_error").fit(X_, y_)
        return self
    
    
    def predict(self, date, scraped_data = None):
        ## If we are including the recent scraped data, append it into our data frame
        ## for better predictions
        if scraped_data is not None:
            bool_vec = np.zeros(len(scraped_data), dtype = bool)
            for i in range(len(scraped_data)):
                scraped_date = scraped_data.iloc[i]["date"]
                if scraped_date not in self.df["date"].values:
                    bool_vec[i] = True
            self.df = pd.concat([self.df, scraped_data[bool_vec]])
            self.df = self.df.sort_values(by = "date")
            
        ## check if date is string
        if isinstance(date, str):
            date = pd.to_datetime(date)
        ## check if date is datetime
        if not isinstance(date, pd.Timestamp):
            raise ValueError("Date must be a string or a pd.Timestamp object") 
        
        ## Clearly, if date is already present in dataframe, we can just return the stored values
        if date in self.df["date"].values:
            return self.df[self.df.date == date][self.to_lag_columns].to_numpy()
        else:
            ## check if last lagged days are in the dataframe
            prior_date_range = pd.date_range(end = date - pd.Timedelta(days=1), periods = self.lag + 1, freq="D")
            for day in prior_date_range:
                
                if day not in self.df["date"].values:
                    _ = self.predict(date=day)
                    
                    
            ## Now, we can predict the values for the date after previous lagged day values are present or predicted
            new_X_ = (
                self.df[(self.df.date >= prior_date_range[0]) & (self.df.date <= prior_date_range[-1])]
                .copy()
            )
                                    
            lagged_new_X = (
                allLaggedDataAvail(df = new_X_, lag = self.lag, to_lag_columns = self.to_lag_columns)
                .drop(columns = self.to_lag_columns)
                .drop(columns = ["date", "location"])
            )            
            
            try:
                predicted_y = self.model.predict(self.scaler.transform(lagged_new_X.to_numpy()))[-1]
            except:
                ## fill missing values in lagged_new_X with 0
                lagged_new_X = lagged_new_X.fillna(0)
                predicted_y = self.model.predict(self.scaler.transform(lagged_new_X.to_numpy()))[-1]
            
            ## clip precipitation to be min 0
            if predicted_y[3] < 0:
                predicted_y[3] = 0
                
            ## clip visibility to be max 10
            if predicted_y[8] > 10:
                predicted_y[8] = 10
            
            ## 
            predicted_obs = pd.DataFrame(predicted_y.reshape(1,10), columns = self.to_lag_columns)
            predicted_obs["date"] = date
            predicted_obs["location"] = self.location
            self.df = pd.concat([self.df, predicted_obs])
            self.df = self.df.sort_values(by = "date").reset_index(drop=True)
            return predicted_y
                    
    def __str__(self):
        return f"{self.location} Model"
    

class WeatherForecast():
    def __init__(self, folder_path: str, lag: int):
        self.df = getCleanDataFrame(folder_path)
        self.locations = self.df.location.unique()
        self.models = {location: LocationModel(location, self.df[self.df.location == location], lag).fit() for location in self.locations}
        
    def getCurrenHourlyWeatherData(self, place: str):
        col_names = ['date', 'Time', 'wind', 'visibility', 'Weather', 'Sky Condition', 'temp', 'dew_point', 'Max Temp 6hr', 
                    'Min Temp 6hr', 'Rel Humidity', 'Wind Chill', 'Heat Index', 'sea_level_pressure', 'Other Pressure', 
                    'precip_1hr', 'Precip 3hr', 'Precip 6hr']
        url = f"https://forecast.weather.gov/data/obhistory/{place}.html"
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_ = 'obs-history')
        for weather_data in table.find_all('tbody'):
            rows = weather_data .find_all('tr')
            data_dict = {col: [] for col in col_names}
            
            for row in rows:
                for col in col_names:
                    data_dict[col].append(row.find_all('td')[col_names.index(col)].text)
            val_list = []
            for i in data_dict["wind"]:
                val = i.split("\n")[1].replace(" ", "").replace("\n", "")
                if val == "":
                    val = "0"
                val_list.append(val)
            data_dict["wind"] = val_list

            for key in data_dict:
                try:
                    data_dict[key] = np.array(data_dict[key], dtype = float)
                except:
                    data_dict[key] = np.array(data_dict[key]) 
        df = (
            pd.DataFrame(data_dict)
            .drop(columns = ["Time", 
                             "Weather", 
                             "Sky Condition", 
                             "Max Temp 6hr", 
                             "Min Temp 6hr", 
                             "Wind Chill", 
                             "Heat Index", 
                             "Other Pressure", 
                             "Precip 3hr", 
                             "Precip 6hr", 
                             "Rel Humidity"])
        )
        ## replace missing values in precip_1hr with 0
        df["precip_1hr"] = df["precip_1hr"].replace("",0).astype(float)
        date_orderings = np.unique(data_dict["date"], return_index = True)
        idxs = np.argsort(np.argsort(date_orderings[1]))
        
        ## for all columns in df, check if there is a missing value, if so, fill it with the previous value
        for col in df.columns:
            try:
                df[col] = df[col].replace("", np.nan).fillna(method = "ffill").fillna(0)
            except:
                pass
        
        ## necessary just in case the change of date happens....date seems to recorded in weather.gov as just the day number
        date_dict = {float(date): int(idxs[i]) for i, date in enumerate(date_orderings[0])}
        return df, date_dict


    def getCurrentWeatherData(self, place: str):
        today = datetime.today()
        hourly_df, date_orderings = self.getCurrenHourlyWeatherData(place)
        daily_df = (
        hourly_df
        .groupby('date')
        .agg({'temp': ['min', 'max', 'mean'],
            'dew_point': ['min', 'max', 'mean'],
            'precip_1hr': 'sum',
            'sea_level_pressure' : 'mean', 
            'visibility': 'max',
            'wind': "max"})
        )
        daily_df.columns = ['_'.join(col) for col in daily_df.columns]
        ## removing grouping of daily_df
        daily_df = (
            daily_df
            .reset_index()
            .rename(columns = {'precip_1hr_sum': 'precipitation',
                               'sea_level_pressure_mean': 'sea_level_pressure',
                               'visibility_max': 'visibility',
                               'wind_max': 'max_wind_speed'})
        )
        
        date_range = pd.date_range(periods = len(daily_df), end = today, freq = "D")[::-1]
        daily_df["date"] = daily_df["date"].apply(lambda x: date_range[date_orderings[x]].floor("1d"))
        daily_df["location"] = place
        daily_df = daily_df.reindex(columns = ['location', 
                                               'date', 
                                               'temp_max', 
                                               'temp_min', 
                                               'temp_mean', 
                                               'precipitation',
                                               'dew_point_max',
                                               'dew_point_min',
                                               'dew_point_mean', 
                                               'max_wind_speed', 
                                               'visibility',
                                               'sea_level_pressure'])
        return daily_df

        
    def predict_location(self, date, location: str):
        try:
            current_data = self.getCurrentWeatherData(location)
        except:
            current_data = None
            
        model = self.models[location]
        _ = model.predict(date, current_data)
        start_date = date - pd.Timedelta(days = 4)
        responses = (
            model.df[(model.df.date >= start_date) & (model.df.date <= date)]
            .copy()[["date", "temp_min", "temp_mean", "temp_max"]]
            .reset_index(drop = True)
            .sort_values(by = "date")        
            .drop(columns = "date")
            .to_numpy()
        )
        return responses
    
    def predict_all(self):
        today = datetime.today().strftime("%Y-%m-%d")
        datetime_today = pd.to_datetime(today)
        datetime_end = datetime_today + timedelta(days = 5)
        
        ## Will need to change some of the codes...
        ## Chicago is KMDW
        ## NY is KSWF
        
        locations_dict = {
            "Anchorage": "PANC",
            "Boise": "KBOI",
            "Chicago": "KORD",
            "Denver": "KCOS", ## Because KDEN is not in the data
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
        #order alphabetically
        locations = sorted(locations_dict.keys())
        #for location in locations:
        predictions = np.zeros((len(locations), 5, 3))
        for i, location in enumerate(locations):
            code = locations_dict[location]
            predictions[i,:,:] = self.predict_location(datetime_end, code)
        predictions = np.round(predictions, 1)
        print(f"{today}, {', '.join(str(ele) for ele in predictions.flatten())}")
        predictions = np.round(predictions, 1)