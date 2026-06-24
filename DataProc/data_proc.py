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
                print(f"Warning: file not exist - {filename}")
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

def un_gz(file_path):
    """ungz zip file"""
    if file_path.suffix != '.gz':
        return
    f_name = str(file_path.with_suffix(''))
    with gzip.open(file_path, 'rb') as g_file:
        with open(f_name, 'wb+') as out_file:            
            shutil.copyfileobj(g_file, out_file)

def quality_check(isd_meter_file):    
    if len(isd_meter_file) <300:
        return 0

def dataproce(f_name, f_path, stationInfo):    
    file_full_path = Path(f_path) / f_name

    try:        
        colspecs = [(0, 4), (5, 7), (8, 10), (11, 13)] + [(i, i + 6) for i in range(13, 56, 6)]
        names = ['year', 'month', 'day', 'hour', 'temp', 'dew-pointtemperature',
                 'pressure', 'winddirection', 'windspeed', 'cloudamount',
                 'hour1ofrainfall', 'hour6ofrainfall']        
        Meter = pd.read_fwf(file_full_path, colspecs=colspecs, names=names, dtype=str, na_values=['-9999'])        
        quality = quality_check(Meter)
        if quality == 0:
            return pd.DataFrame()

    except Exception as e:
        print(f"Read file {file_full_path} failed: {e}")
        return pd.DataFrame()   
    numeric_cols = ['temp', 'dew-pointtemperature', 'pressure', 'winddirection',
                    'windspeed', 'cloudamount', 'hour1ofrainfall', 'hour6ofrainfall']

    for col in numeric_cols:
        if col in Meter.columns:
            Meter[col] = pd.to_numeric(Meter[col], errors='coerce')
    
    precip_cols = ['hour1ofrainfall', 'hour6ofrainfall']
    for col in precip_cols:
        if col in Meter.columns:            
            Meter.loc[Meter[col] == -1, col] = 0.05
            Meter.loc[Meter[col] < 0, col] = 0    
    try:        
        stem = Path(f_name).stem
        parts = stem.split('-')
        if len(parts) < 3:
            raise ValueError("File name's format wrong!")
        stationid_6digit = parts[0]        
        mask = stationInfo['STATION_ID'].astype(str).str.startswith(stationid_6digit)
        matching_rows = stationInfo[mask]

        if matching_rows.empty:
            print(f"Warning: ID {stationid_6digit} not found!")            
            station_id_val = stationid_6digit
            lat_val = lon_val = elev_val = greenup_val = np.nan
        else:            
            row = matching_rows.iloc[0]
            station_id_val = str(row['STATION_ID']) 
            # station_id_val = stationid_6digit
            lat_val = row['LATITUDE']
            lon_val = row['LONGITUDE']
            elev_val = row['ELEVATION']            
            year_val = int(Meter['year'].iloc[0])
            greenup_col = f"Greenup_{year_val}"
            
            if greenup_col in row.index:
                greenup_val = row[greenup_col]                
                if greenup_val == 32767:                   
                    return pd.DataFrame()
                else:
                    date = datetime(1970, 1, 1) + timedelta(days=greenup_val)
                    greenup_val = date.timetuple().tm_yday
            else:                
                greenup_val = np.nan
                return pd.DataFrame()

    except Exception as e:        
        return pd.DataFrame()
    valid_records = len(Meter.dropna(subset=['temp']))
    if valid_records < 300:        
        return pd.DataFrame()
    
    SCALE_FACTOR = 10
    base_cols = ['year', 'month', 'day']
    agg_dict = {
        'temp': ['max', 'min', 'mean'],
        'pressure': ['mean'],
        'hour6ofrainfall': ['sum'],
        'windspeed': ['mean'],
        'cloudamount': ['mean'],
        'dew-pointtemperature': ['mean']
    }    
    agg_dict = {k: v for k, v in agg_dict.items() if k in Meter.columns}
    if not agg_dict:        
        return pd.DataFrame()   
    daily_stats = Meter.groupby(base_cols).agg(agg_dict).reset_index()    
    daily_stats.columns = ['_'.join(col).strip('_') for col in daily_stats.columns]
    
    rename_dict = {
        'temp_max': 'Temp_max',
        'temp_min': 'Temp_min',
        'temp_mean': 'Temp_mean',
        'pressure_mean': 'VaporP_mean',
        'hour6ofrainfall_sum': 'precipitation_sum',
        'windspeed_mean': 'windspeed_mean',
        'cloudamount_mean': 'cloudamount',
        'dew-pointtemperature_mean': 'dew-pointtemperature'
    }
    daily_stats = daily_stats.rename(columns=rename_dict)    
    scale_cols = ['Temp_max', 'Temp_min', 'Temp_mean', 'VaporP_mean',
                  'precipitation_sum', 'windspeed_mean', 'cloudamount', 'dew-pointtemperature']
    for col in scale_cols:
        if col in daily_stats.columns:
            daily_stats[col] = (daily_stats[col] / SCALE_FACTOR).round(2)
    
    year_val = int(daily_stats['year'].iloc[0])
    is_leap = (year_val % 4 == 0 and year_val % 100 != 0) or (year_val % 400 == 0)
    days_in_year = 366 if is_leap else 365    
    date_range = pd.date_range(start=f"{year_val}-01-01", periods=days_in_year, freq='D')    
    daily_stats['date'] = pd.to_datetime(daily_stats[['year', 'month', 'day']])
    
    daily_stats = pd.DataFrame({'date': date_range}).merge(daily_stats, on='date', how='left')
   
    daily_stats['STATION_ID'] = station_id_val
    daily_stats['LATITUDE'] = lat_val
    daily_stats['LONGITUDE'] = lon_val
    daily_stats['ELEVATION'] = elev_val
    daily_stats['Gdoy'] = greenup_val    
    daily_stats['year'] = daily_stats['date'].dt.year
    daily_stats['month'] = daily_stats['date'].dt.month
    daily_stats['day'] = daily_stats['date'].dt.day
    daily_stats['DOY'] = daily_stats['date'].dt.dayofyear
    
    numeric_cols_to_fill = ['Temp_max', 'Temp_min', 'Temp_mean', 'VaporP_mean',
                            'precipitation_sum', 'windspeed_mean', 'cloudamount', 'dew-pointtemperature']
    for col in numeric_cols_to_fill:
        if col in daily_stats.columns:            
            daily_stats[col] = daily_stats[col].interpolate(method='linear', limit_direction='both')            
            daily_stats[col] = daily_stats[col].fillna(method='ffill').fillna(method='bfill')
   
    import math
    def calc_photoperiod(row):
        lat = row['LATITUDE']
        doy = row['DOY']
        
        sin1 = 2 * math.pi * doy / 365 - 1.39
        tan1 = math.tan(0.409 * math.sin(sin1))
       
        tan2 = -math.tan(lat * math.pi / 180)        
        acos_arg = max(-1.0, min(1.0, tan2 * tan1))
       
        photoperiod = (24 / math.pi) * math.acos(acos_arg)
        return round(photoperiod, 2)

    daily_stats['PHO'] = daily_stats.apply(calc_photoperiod, axis=1)
    cols = ['STATION_ID', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'year', 'month','day','DOY',
            'Temp_max', 'Temp_min', 'Temp_mean',  'VaporP_mean', 'precipitation_sum',
            'dew-pointtemperature', 'windspeed_mean', 'PHO', 'cloudamount', 'Gdoy']

    cols = [c for c in cols if c in daily_stats.columns]
    daily_stats = daily_stats[cols]

    return daily_stats

def ExtrMeteoData(ISD_MeteoroData_path):
    stationInfo = pd.read_csv(data_path / "StationInfo.csv")    
    stationInfo['STATION_ID'] = pd.to_numeric(stationInfo['STATION_ID'], errors='coerce')    
    valid_years = set(map(str, range(2001, 2023)))
    
    df_list, y_Greenup, y_Dormancy = [], [], []
    base_path = Path(ISD_MeteoroData_path)
    output_path = data_path / "StationMetePheno.parquet"
    has_data = False
    
    for year_folder in base_path.iterdir():        
        if not year_folder.is_dir():
            continue
        year_str = year_folder.name        
        if year_str not in valid_years:
            continue
       
        target_files_map = {}        
        for sid in stationInfo['STATION_ID'].dropna():            
            sid_str = str(sid).zfill(6)            
            expected_filename = f"{sid_str[:6]}-{sid_str[6:]}-{year_str}.gz"           
            target_files_map[expected_filename] = sid           
            file_path = year_folder / expected_filename
            
            if not file_path.exists():                
                continue
            station_id = sid
            try:                
                un_gz(file_path)
                
                file_name_no_ext = file_path.stem               
                meter_data = dataproce(file_name_no_ext, str(year_folder), stationInfo)
               
                extracted_file = year_folder / file_name_no_ext
                if extracted_file.exists():
                    extracted_file.unlink()

                if meter_data.empty:                    
                    continue 
                    
                df_list.append(meter_data)               

            except Exception as e:                
                continue        

    if df_list:        
        final_df = pd.concat(df_list, ignore_index=True)        
        final_df.to_parquet(
                output_path,
                engine='pyarrow',
                compression='snappy'
        )
        
