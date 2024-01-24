import psycopg2
import pandas as pd
import numpy as np
import json

def all_info_from_sql(data_dir):
    def speed_direction_temperature(data_dir=None):
        with open(f"{data_dir}/domecam/wind_curr2.json") as f:
            data = json.load(f)
            tmp = data.get('results')[0].get('series')[0].get('values')
            all_wind_speed = []
            all_temerature = []
            all_wind_direction = []
            for item in tmp:
                if item[-1] == 'WIND':
                    print('wind speed:', item[2])
                    all_wind_speed.append(item[2])
                if item[-1] == 'WIND_DIR':
                    print('wind direction:', item[2])
                    all_wind_direction.append(item[2])
                if item[-1] == 'TEMP_EX':
                    print('temperature:', item[2])
                    all_temerature.append(item[2])
            result_wind_speed = np.mean(all_wind_speed)
            result_wind_direction = np.mean(all_wind_direction) 
            result_temperature = np.mean(all_temerature)

        return result_temperature, result_wind_direction, result_wind_speed

    def telescope_temerarute():
        conn = psycopg2.connect(
            host="192.168.10.87",
            database="enviroment",
            user="tdsuser",
            password="?tdsuser=")
        cur = conn.cursor()

        cur.execute("SELECT ts_id,meas_time,value from \"sai2p5_temp\" WHERE (meas_time> now() - INTERVAL '3 HOUR' - INTERVAL '1 MINUTE') AND ((ts_id=1) or (ts_id=2) or (ts_id=3) or (ts_id=4) or (ts_id=5) or (ts_id=6) or (ts_id=7) or (ts_id=8) or (ts_id=9) or (ts_id=10) or (ts_id=11) or (ts_id=12) or (ts_id=13) or (ts_id=14) or (ts_id=15) or (ts_id=16) or (ts_id=17) or (ts_id=19));")

        data = cur.fetchall()

        ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

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
            if item[0] == 10:
                ts_10.append(item[2])
            if item[0] == 11:
                ts_11.append(item[2])
            if item[0] == 12:
                ts_12.append(item[2])
            if item[0] == 13:
                ts_13.append(item[2])
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
        return ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19
    
    result_temperature, result_wind_direction, result_wind_speed = speed_direction_temperature(data_dir=data_dir)
    ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19 = telescope_temerarute()
    return result_temperature, result_wind_direction, result_wind_speed, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19