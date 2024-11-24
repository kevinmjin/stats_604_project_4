from weather_prediction.util import WeatherForecast, LocationModel
import os, pickle, sys
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)

## can modify this to use a different lag depending on the model
lag = int(sys.argv[1])
WFobj = WeatherForecast("data/", lag)
# save test
os.makedirs("models", exist_ok=True)
with open(f"models/WFobj_{lag}.pkl", "wb") as f:
    pickle.dump(WFobj, f)