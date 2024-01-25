import psycopg2
import pandas as pd
import numpy as np
import json
import os
import warnings
import subprocess

from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz

def all_info_from_sql(data_dir, file_name, file, file_time, file_time_ub):
    def speed_direction_temperature(data_dir, file_time, file_time_ub):
        cmd_sql = f"curl -G -H \"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNhZm9ub3YiLCJleHAiOjE3NjcyNTgwMDB9.GZ6_LQfb1L_kZNtF4z8Zrf8IgRD9N9DRwC2eEfR9bmQ\" 'http://eagle.sai.msu.ru/query?pretty=true' --data-urlencode \"db=collectd\" --data-urlencode \"q=select * from collectd.archive./collectd-sv.plugin_value/ where time>'{file_time}' and time<'{file_time_ub}';\" > wind_curr2.json"
        os.system(cmd_sql)
        
        with open(f"{data_dir}/domecam/wind_curr2.json") as f:
            data = json.load(f)
            tmp = data.get('results')[0].get('series')[0].get('values')
            all_wind_speed = []
            all_temerature = []
            all_wind_direction = []
            for item in tmp:
                if item[-1] == 'WIND':
                    all_wind_speed.append(item[2])
                if item[-1] == 'WIND_DIR':
                    all_wind_direction.append(item[2])
                if item[-1] == 'TEMP_EX':
                    all_temerature.append(item[2])
            result_wind_speed = np.mean(all_wind_speed)
            result_wind_direction = np.mean(all_wind_direction) 
            result_temperature = np.mean(all_temerature)

        return result_temperature, result_wind_direction, result_wind_speed

    def telescope_temerarute(file_time=None, file_time_ub=None):
        conn = psycopg2.connect(
            host="192.168.10.87",
            database="enviroment",
            user="tdsuser",
            password="?tdsuser=")
        cur = conn.cursor()
        cmd_sql_execute = f"SELECT ts_id,meas_time,value from \"sai2p5_temp\" WHERE ( meas_time>'{file_time}' and meas_time<'{file_time_ub}') AND ((ts_id=1) or (ts_id=2) or (ts_id=3) or (ts_id=4) or (ts_id=5) or (ts_id=6) or (ts_id=7) or (ts_id=8) or (ts_id=9) or (ts_id=11) or (ts_id=12) or (ts_id=14) or (ts_id=15) or (ts_id=16) or (ts_id=17) or (ts_id=19));"
        cur.execute(cmd_sql_execute)

        data = cur.fetchall()

        ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_11, ts_12, ts_14, ts_15, ts_16, ts_17, ts_19 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for item in data:
            if item[0] == 1:
                ts_1.append(item[2])
            if item[0] == 2:
                ts_2.append(item[2])
            if item[0] == 3:
                ts_3.append(item[2])
            if item[0] == 4:
                ts_4.append(item[2])
            if item[0] == 5:
                ts_5.append(item[2])
            if item[0] == 6:
                ts_6.append(item[2])
            if item[0] == 7:
                ts_7.append(item[2])
            if item[0] == 8:
                ts_8.append(item[2])
            if item[0] == 9:
                ts_9.append(item[2])
            if item[0] == 11:
                ts_11.append(item[2])
            if item[0] == 12:
                ts_12.append(item[2])
            if item[0] == 14:
                ts_14.append(item[2])
            if item[0] == 15:
                ts_15.append(item[2])
            if item[0] == 16:
                ts_16.append(item[2])
            if item[0] == 17:
                ts_17.append(item[2])
            if item[0] == 19:
                ts_19.append(item[2])

        cur.close()
        conn.close()
        warnings.simplefilter("ignore")
        return np.mean(ts_1), np.mean(ts_2), np.mean(ts_3), np.mean(ts_4), np.mean(ts_5), np.mean(ts_6), np.mean(ts_7), np.mean(ts_8), np.mean(ts_9), np.mean(ts_11), np.mean(ts_12), np.mean(ts_14), np.mean(ts_15), np.mean(ts_16), np.mean(ts_17), np.mean(ts_19)
    
#     def telescope_coords(file_time, ra, dec):
#         observing_location = EarthLocation(lat='43 44 10', lon='42 40 03', height=2100*u.m)
#         observing_time = Time(file_time)  
#         aa = AltAz(location=observing_location, obstime=observing_time)
#         coord = SkyCoord(ra, dec)
#         cAltAz = coord.transform_to(aa)
#         return cAltAz.alt.deg, cAltAz.az.deg

    result_temperature, result_wind_direction, result_wind_speed = speed_direction_temperature(data_dir, file_time, file_time_ub)
    ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_11, ts_12, ts_14, ts_15, ts_16, ts_17, ts_19 = telescope_temerarute(file_time=file_time, file_time_ub=file_time_ub)
    
    all_info = [round(result_temperature, 2), round(result_wind_direction, 2), round(result_wind_speed, 2), ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_11, ts_12, ts_14, ts_15, ts_16, ts_17, ts_19]
    df = pd.DataFrame([all_info], columns = ['temperute', 'wind direction', 'wind speed', 'ts_1', 'ts_2', 'ts_3', 'ts_4', 'ts_5', 'ts_6', 'ts_7', 'ts_8', 'ts_9', 'ts_11', 'ts_12', 'ts_14', 'ts_15', 'ts_16', 'ts_17', 'ts_19'])
    df.to_csv(f'{data_dir}/results/{file_name}/{file[:-5]}_info_from_logs.txt', index=False)
    
    
    return result_temperature