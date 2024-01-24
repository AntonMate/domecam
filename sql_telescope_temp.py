import psycopg2
conn = psycopg2.connect(
    host="192.168.10.87",
    database="enviroment",
    user="tdsuser",
    password="?tdsuser=")
cur = conn.cursor()

cur.execute("SELECT ts_id,meas_time,value from \"sai2p5_temp\" WHERE (meas_time> now() - INTERVAL '3 HOUR' - INTERVAL '1 MINUTE') AND ((ts_id=1) or (ts_id=2) or (ts_id=3) or (ts_id=4) or (ts_id=5) or (ts_id=6) or (ts_id=7) or (ts_id=8) or (ts_id=9) or (ts_id=10) or (ts_id=11) or (ts_id=12) or (ts_id=13) or (ts_id=14) or (ts_id=15) or (ts_id=16) or (ts_id=17) or (ts_id=18) or (ts_id=19));")

data = cur.fetchall()
for  k in range(1, 20, 1):
    for item in data:
        if item[0] == k:
            print(f'TS-{k}:', item[2])
cur.close()
conn.close()