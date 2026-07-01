import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from joblib import load
from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.kernel_based import RocketRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

#####################################################################################################################
# Utility Functions: Prepare data in the format specified by the model
#####################################################################################################################
def Format_sktime(data):
    """
    Generate the 3D data required for both KNN and Rocket  
    :param data:
    :return: x_KNN_TS, x_Rocket_TS, y_sktime_TS
    """
    data_KNN_TS = data.copy()
   
    # Set the DOY within each site-year to 1~winLength
    def reset_doy(df):
        df = df.copy()
        df['DOY'] = np.arange(1, len(df) + 1)  # Increment starting from 1
        return df

    data_KNN_TS = data_KNN_TS.groupby(['STATION_ID', 'year'], group_keys=False).apply(reset_doy)   

   

