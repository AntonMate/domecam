from main import processDomecam
import os
import getopt
import sys
from datetime import datetime, timedelta
from sql_telescope_temp import all_info_from_sql

# ============================================================================
latency = [4, 6] # задержка для кросс-корреляции, [кадр]
conjugated_distance = 2 # сопряженное расстояние, [километр]
D = 2.5 # диаметр телескопа, [метр]
spectrum = 'poly' # тип излучения
lambda_ = 650*1e-9 # длина наблюдаемой волны света, [метр]

# для полихроматического излучения нужны кривые фильтра, детектора и звезды
file_filter = 'KC19_d16t4_Safonov.xls'
file_ccd = 'ccd_prosilica2.crv'
# file_star = 'a05.sp'

use_gradient = False # БС: использовать ли градиенты между слоями
use_windvar = True # использовать ли дисперсию ветра

# БС: если параметр do_fitting равен False: отладочный режим, аппроксимация не будет выполнена, а будут взяты начальные 
# параметры initial_params
# БС: если параметр do_fitting равен True, то будет выполнена оценка начальных параметров и проведена аппроксимация
do_fitting = True
dome_only = 5 # 0, чтобы отключить. >0, чтобы задать радиус области вокруг центра
input_parametrs = [[0, 0, 2*0.5495173, 2, 0.25]]
# ============================================================================
if do_fitting == True:
    initial_params = None
else:
    initial_params = input_parametrs
# начальные параметры для аппроксимации в явном виде
# если их не указать, то они будут подобраны автоматически
# ============================================================================
optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
new_path = optlist[0][1]

if new_path.endswith('.fits'):
    print(f' - MODE: обработка одного файла {new_path}')
    indexes = [i for i in range(len(new_path)) if new_path[i] == "/"]
    data_dir = new_path[:indexes[-1]]
    print(' - dir:', data_dir)
    file = new_path[indexes[-1]+1:] # полное название фаайла серии
    print(' -', file)
    file_name = file.replace('_2km.fits', '')
    file_name = file_name.replace('_0km.fits', '') # номер серии
    
    file_time = file_name.replace('DC', '') # дата записи из названия файла
    file_time = datetime.strptime(file_time, '%y%m%d%H%M%S')
    file_time_ub = file_time + timedelta(minutes=5)
    cmd_sql = f"curl -G -H \"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNhZm9ub3YiLCJleHAiOjE3NjcyNTgwMDB9.GZ6_LQfb1L_kZNtF4z8Zrf8IgRD9N9DRwC2eEfR9bmQ\" 'http://eagle.sai.msu.ru/query?pretty=true' --data-urlencode \"db=collectd\" --data-urlencode \"q=select * from collectd.archive./collectd-sv.plugin_value/ where time>'{file_time}' and time<'{file_time_ub}';\" > wind_curr2.json"
    os.system(cmd_sql)  
    
    result_temperature, result_wind_direction, result_wind_speed, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19 = all_info_from_sql(data_dir=data_dir)
    
    print(result_temperature, result_wind_direction, result_wind_speed, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19)
    
    indexes_h = [i for i in range(len(file)) if file[i] == "k"] # file[indexes_h[0]-1] высота сопряжения
    for item in os.listdir(data_dir):
        if 'bias' in item and file_name in item and f'{file[indexes_h[0]-1]}km' in item:
            print(' - bias:', item)
            file_bias = item

    with open('logs.txt') as f:
        for line in f:
            if file in line:
                print(' - logs.txt:', line.strip())
                star_name = line.split()[1]
                file_star = f'{line.split()[2].lower()}.sp'
                print(f' - spectrum: {file_star}')
    
    processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=initial_params, dome_only=dome_only, use_windvar=use_windvar, star_name=star_name, latency_list=latency)
                
else:
    print(f' - MODE: обработка всех файлов _2km.fits из {new_path}')
    data_dir = new_path
    for ser in os.listdir(new_path):
        if ser.endswith('_2km.fits'):
            print('')
            file = ser # номер серии
            print(' -', file)
            file_name = file.replace('_2km.fits', '') # номер серии
            
            file_time = file_name.replace('DC', '') # дата записи из названия файла
            file_time = datetime.strptime(file_time, '%y%m%d%H%M%S')
            file_time_ub = file_time + timedelta(minutes=5)
            cmd_sql = f"curl -G -H \"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNhZm9ub3YiLCJleHAiOjE3NjcyNTgwMDB9.GZ6_LQfb1L_kZNtF4z8Zrf8IgRD9N9DRwC2eEfR9bmQ\" 'http://eagle.sai.msu.ru/query?pretty=true' --data-urlencode \"db=collectd\" --data-urlencode \"q=select * from collectd.archive./collectd-sv.plugin_value/ where time>'{file_time}' and time<'{file_time_ub}';\" > wind_curr2.json"
            os.system(cmd_sql)  
            
            result_temperature, result_wind_direction, result_wind_speed, ts_1, ts_2, ts_3, ts_4, ts_5, ts_6, ts_7, ts_8, ts_9, ts_10, ts_11, ts_12, ts_13, ts_14, ts_15, ts_16, ts_17, ts_19 = all_info_from_sql(data_dir=data_dir)
            
            indexes_h = [i for i in range(len(file)) if file[i] == "k"] # file[indexes_h[0]-1] высота сопряжения
            for item in os.listdir(data_dir):
                if 'bias' in item and file_name in item and f'{file[indexes_h[0]-1]}km' in item:
                    print(' - bias:', item)
                    file_bias = item
                    
            with open('logs.txt') as f:
                for line in f:
                    if file in line:
                        print(' - logs.txt:', line.strip())
                        star_name = line.split()[1]
                        file_star = f'{line.split()[2].lower()}.sp'
                        print(f' - spectrum: {file_star}')
            
            processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=initial_params, dome_only=dome_only, use_windvar=use_windvar, star_name=star_name, latency_list=latency)
# ============================================================================

