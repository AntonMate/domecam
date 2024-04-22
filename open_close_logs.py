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

    for item in data:
        if item[0] == 1:
            ts_1.append(item) # item[2]
        if item[0] == 2:
            ts_2.append(item)
        if item[0] == 3:
            ts_3.append(item)
        if item[0] == 17:
            ts_17.append(item)
        if item[0] == 19:
            ts_19.append(item)

    cur.close()
    conn.close()
    warnings.simplefilter("ignore")

    return ts_1, ts_2, ts_3, ts_17, ts_19

df = pd.read_csv("logs_open_close.csv")
lb = df['open']
ub = df['close']

all_ts1 = []
all_ts2 = []
all_ts3 = []
all_ts17 = []
all_ts19 = []

for i in range(4):
    print('doing:', lb[i], ub[i])
    ts1, ts2, ts3, ts17, ts19 = telescope_temperature(file_time=lb[i], file_time_ub=ub[i])
    print(ts1, ts2, ts3, ts17, ts19)
    all_ts1.append(ts1)
    all_ts2.append(ts2)
    all_ts3.append(ts3)
    all_ts17.append(ts17)
    all_ts19.append(ts19)

with open('all_ts1.txt', 'w') as f:
    print(all_ts1, file=f)

with open('all_ts2.txt', 'w') as f:
    print(all_ts2, file=f)

with open('all_ts3.txt', 'w') as f:
    print(all_ts3, file=f)

with open('all_ts17.txt', 'w') as f:
    print(all_ts17, file=f)
    
with open('all_ts19.txt', 'w') as f:
    print(all_ts19, file=f)