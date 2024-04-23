import pandas as pd
import numpy as np
import json
import os
import warnings
import psycopg2

def telescope_temperature(file_time=None, file_time_ub=None):
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
    date_mirror_temperature, date_indoor_temperature = [], []

    for item in data:
        if item[0] == 1:
            ts_1.append(item[2]) # item[2]
            date_mirror_temperature.append(item[1])
        if item[0] == 2:
            ts_2.append(item[2])
        if item[0] == 3:
            ts_3.append(item[2])
        if item[0] == 17:
            ts_17.append(item[2])
            date_indoor_temperature.append(item[1])
        if item[0] == 19:
            ts_19.append(item[2])

    cur.close()
    conn.close()
    warnings.simplefilter("ignore")
    mirror_temperature = (np.asarray(ts_1) + np.asarray(ts_2) + np.asarray(ts_3))/30
    indoor_temperature = (np.asarray(ts_17) + np.asarray(ts_19))/20
    return mirror_temperature, date_mirror_temperature, indoor_temperature, date_indoor_temperature

df = pd.read_csv("logs_open_close.csv")
lb = df['open']
ub = df['close']
ub_lb = df['close_open']

all_mirror_temperature, all_date_mirror_temperature, all_indoor_temperature, all_date_indoor_temperature = [], [], [], []

for i in range(len(df)):
    print(' - doing:', lb[i], '|', ub[i], '|', ub_lb[i])
    mirror_temperature, date_mirror_temperature, indoor_temperature, date_indoor_temperature = telescope_temperature(file_time=lb[i], file_time_ub=ub[i])
    #print(mirror_temperature, date_mirror_temperature, indoor_temperature, date_indoor_temperature)
    print(' - done!')
    all_mirror_temperature.append(mirror_temperature)
    all_date_mirror_temperature.append(date_mirror_temperature)
    all_indoor_temperature.append(indoor_temperature)
    all_date_indoor_temperature.append(date_indoor_temperature)

with open('mirror_temperature.txt', 'w') as f:
    print(all_mirror_temperature, file=f)

with open('date_mirror_temperature.txt', 'w') as f:
    print(all_date_mirror_temperature, file=f)

with open('indoor_temperature.txt', 'w') as f:
    print(all_indoor_temperature, file=f)

with open('date_indoor_temperature.txt', 'w') as f:
    print(all_date_indoor_temperature, file=f)

print('ALL DONE')