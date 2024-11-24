
from weather_prediction.util import WeatherForecast, LocationModel
import pickle, sys
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)

lag = int(sys.argv[1])
with open(f"models/WFobj_{lag}.pkl", "rb") as f:
    wf = pickle.load(f)
predictions = wf.predict_all()