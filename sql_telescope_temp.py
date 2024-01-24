import psycopg2
conn = psycopg2.connect(
    host="192.168.10.87",
    database="enviroment",
    user="tdsuser",
    password="?tdsuser=")
cur = conn.cursor()
cur.execute("SELECT ts_id,meas_time,value from \"sai2p5_temp\" WHERE meas_time> now() - INTERVAL '3 HOUR' - INTERVAL '2 MINUTE';")

data = cur.fetch()
print(data)
cur.close()
conn.close()