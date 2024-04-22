import pandas as pd
import numpy as np
import json
import os
import warnings
import psycopg2

def telescope_temerarute(file_time=None, file_time_ub=None):
    conn = psycopg2.connect(
        host="192.168.10.87",
        database="enviroment",
        user="tdsuser",
        password="?tdsuser=")
    cur = conn.cursor()
    cmd_sql_execute = f"SELECT ts_id,meas_time,value from \"sai2p5_temp\" WHERE ( meas_time>'{file_time}' and meas_time<'{file_time_ub}') AND ((ts_id=1) or (ts_id=2) or (ts_id=3) or (ts_id=17) or (ts_id=19));"
    cur.execute(cmd_sql_execute)

    data = cur.fetchall()

    ts_1, ts_2, ts_3, ts_17, ts_19 = [], [], [], [], []

    for item in data:
        if item[0] == 1:
            ts_1.append(item[2])
        if item[0] == 2:
            ts_2.append(item[2])
        if item[0] == 3:
            ts_3.append(item[2])
        if item[0] == 17:
            ts_17.append(item[2])
        if item[0] == 19:
            ts_19.append(item[2])

    cur.close()
    conn.close()
    warnings.simplefilter("ignore")
    
    mirror_temperature = (np.mean(ts_1)/10 + np.mean(ts_2)/10 + np.mean(ts_3)/10)/3
    indoor_temperuature = (np.mean(ts_17)/10 + np.mean(ts_19)/10)/2
    return mirror_temperature, indoor_temperuature

df = pd.read_csv("logs_open_close.csv")
lb = df['open']
ub = df['close']

all_mirror_temperature = []
all_indoor_temperuature = []

for i in range(len(df)):
    a,b = telescope_temerarute(file_time=lb[i], file_time_ub=ub[i])
    all_mirror_temperature.append(a)
    all_indoor_temperuature.append(b)

with open('all_mirror_temperature', 'w') as f:
    print(all_mirror_temperature, file=f)

with open('all_indoor_temperature', 'w') as f:
    print(all_indoor_temperature, file=f)