import os
import gzip
import shutil
import random
import numpy as np
import pandas as pd
from osgeo import gdal
from pathlib import Path
from datetime import datetime, timedelta

current_dir = Path(__file__).resolve().parent.parent
data_path = current_dir / "data"

def station_extract():
    data = pd.read_csv( data_path / 'isd-history2024.csv')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
  
    neilonbool1 = data['LON'] > 97  # 101
    neilonbool2 = data['LON'] < 128
    neilonbool = neilonbool1 & neilonbool2

    neilatbool1 = data['LAT'] > 37
    neilatbool2 = data['LAT'] < 54  # 47
    neilatbool = neilatbool1 & neilatbool2
    neilatlonbool = neilonbool & neilatbool
    
    begin = data['BEGIN'].apply(lambda x: str(x)[0:4])   
    end = data['END'].apply(lambda x: str(x)[0:4])
    beginbool = begin.apply(lambda x: int(x)) <= 2001
    endbool = end.apply(lambda x: int(x)) >= 2022
    datebool = beginbool & endbool

    bool = neilatlonbool & datebool
    data['STATION_ID'] = data['USAF']
    stationID = (data['STATION_ID'][bool]).apply(lambda x: str(x) + '99999')    
    data['LONGITUDE'] = data['LON']
    stationLon = data['LONGITUDE'][bool]    
    data['LATITUDE'] = data['LAT']
    stationLat = data['LATITUDE'][bool]    
    data['ELEVATION'] = data['ELEV(M)']
    stationElev = data['ELEVATION'][bool]    
    data['STATION NAME'] = data['STATION NAME']
    stationName = data['STATION NAME'][bool]

    stationInfo = pd.concat([stationID, stationName, stationLon, stationLat, stationElev], axis=1, ignore_index=False)   
    # print(len(stationID), len(stationInfo)) #244个无重复站点    
    stationInfo.drop_duplicates(inplace=True, ignore_index=True)
    stationInfo.reset_index(drop=True, inplace=True)   
    stationInfo.to_csv(data_path / 'menggu_stations.csv', index=False) # 用于绘制研究区概况图

def ExtrMCD12Q2():
    phenology = ['Greenup']
    years = range(2001, 2023)    
    stationInfo = pd.read_csv(data_path / 'menggu_stations.csv',dtype={2: float, 3: float})

    for idx_year, year in enumerate(years):
        for f in phenology:
            tifile = f'MCD12Q2.A{year}001.{f}.Num_Modes_01.tif'
            filename = data_path / tifile

            if not filename.exists():
                print(f"警告: 文件不存在 - {filename}")
                continue

            dataset = gdal.Open(str(filename))
            if dataset is None:
                continue           
            geotrans = dataset.GetGeoTransform()
            band = dataset.GetRasterBand(1)           
            array_data = band.ReadAsArray()
            
            lons = stationInfo.iloc[:, 2].values
            lats = stationInfo.iloc[:, 3].values
            
            x_offsets = np.round((lons - geotrans[0]) / geotrans[1]).astype(int) 
            y_offsets = np.round((lats - geotrans[3]) / geotrans[5]).astype(int)            
            valid_mask = (x_offsets >= 0) & (x_offsets < array_data.shape[1]) & \
                         (y_offsets >= 0) & (y_offsets < array_data.shape[0])
            
            extracted_values = np.where(valid_mask,
                                        array_data[y_offsets[valid_mask], x_offsets[valid_mask]],
                                        np.nan)            
            col_name = f'{phenology[0]}_{year}'
            stationInfo[col_name] = extracted_values
            del array_data  
            dataset = None
    stationInfo.to_csv(data_path / "StationInfo.csv", index=False)
    
