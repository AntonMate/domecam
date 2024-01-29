from main import processDomecam
import os
import getopt
import sys
from datetime import datetime, timedelta
from sql_telescope_temp import all_info_from_sql

from astropy.io import fits


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
do_crosscorr = False # подсчет кросс корреляции

# БС: если параметр do_fitting равен False: отладочный режим, аппроксимация не будет выполнена, а будут взяты начальные 
# параметры initial_params
# БС: если параметр do_fitting равен True, то будет выполнена оценка начальных параметров и проведена аппроксимация
do_fitting = False
dome_only = 5 # 0, чтобы отключить. >0, чтобы задать радиус области вокруг центра
input_parametrs = [[0, 0, 2*0.5495173, 2, 0.25]]

err_files = ['DC220131123749_2km.fits', 'DC220506144552_2km.fits'] # файлы/серии, которые нужно исключить общей обработки

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
    print('')
    print(f' - MODE: fitting {new_path}')
    indexes = [i for i in range(len(new_path)) if new_path[i] == "/"]
    data_dir = new_path[:indexes[-1]]
    print(' - dir:', data_dir)
    file = new_path[indexes[-1]+1:] # полное название фаайла серии
    file_name = file.replace('_2km.fits', '')
    file_name = file_name.replace('_0km.fits', '') # номер серии
    
    # создание папки, куда будут сохраняться результаты
    if not os.path.isdir(f'{data_dir}/results'):
        os.mkdir(f'{data_dir}/results')
    # создание папки с результатами обработки серии
    if not os.path.isdir(f'{data_dir}/results/{file_name}'):
        os.mkdir(f'{data_dir}/results/{file_name}')
    
    file_time = file_name.replace('DC', '') # дата записи из названия файла
    file_time = datetime.strptime(file_time, '%y%m%d%H%M%S')
    file_time_ub = file_time + timedelta(minutes=5)  
    
    result_temperature = all_info_from_sql(data_dir, file_name, file, file_time, file_time_ub) # всю инофрмацию из базы данных в дополнительный файл txt пишу, чтобы не загромождать
     
    indexes_h = [i for i in range(len(file)) if file[i] == "k"] # file[indexes_h[0]-1] высота сопряжения
    
    file_bias = None
    metka_bias = 'found'
    for item in os.listdir(data_dir):                
        if 'bias' in item and file_name in item and f'{file[indexes_h[0]-1]}km' in item:
            file_bias = item
            print(f' - {file} --> bias: {item}')
            
    if file_bias is None:
        if file == 'DC231008152952_2km.fits':
            file_bias = 'DC230902202035_2km_bias.fits'
            metka_bias = 'DC230902202035_2km_bias.fits'
        else:
            if file.startswith('DC2202') or file.startswith('DC2203') or file.startswith('DC2205') or file.startswith('DC2211'):
                file_bias = 'DC221108182951_2km_bias.fits'
                metka_bias = 'DC221108182951_2km_bias.fits'
        print(f' - {file} --> bias: not found, took bias: {file_bias}')
            
    with open('logs2.txt') as f:
        for line in f:
            if file in line:
                star_name = line.split()[1]
                file_star = f'{line.split()[2].lower()}.sp'
                alt = round(float(line.split()[-2]), 4)
                az = round(float(line.split()[-1]), 4)
                print(f' - logs.txt: {line.strip()} --> spectrum: {file_star}')
    
    processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=initial_params, dome_only=dome_only, use_windvar=use_windvar, star_name=star_name, latency_list=latency, alt=alt, az=az, do_crosscorr=do_crosscorr, metka_bias=metka_bias)
                
else:
    print('')
    print(f' - MODE: fitting all files _2km.fits from {new_path}')
    data_dir = new_path
    for ser in os.listdir(new_path):
        if ser.endswith('_2km.fits') and ser not in err_files:
            print('')
            file = ser # номер серии
            file_name = file.replace('_2km.fits', '') # номер серии
            
            # создание папки, куда будут сохраняться результаты
            if not os.path.isdir(f'{data_dir}/results'):
                os.mkdir(f'{data_dir}/results')
            # создание папки с результатами обработки серии
            if not os.path.isdir(f'{data_dir}/results/{file_name}'):
                os.mkdir(f'{data_dir}/results/{file_name}')
      
            file_time = file_name.replace('DC', '') # дата записи из названия файла
            file_time = datetime.strptime(file_time, '%y%m%d%H%M%S')
            file_time_ub = file_time + timedelta(minutes=5)
            
            result_temperature = all_info_from_sql(data_dir, file_name, file, file_time, file_time_ub)
            
            indexes_h = [i for i in range(len(file)) if file[i] == "k"] # file[indexes_h[0]-1] высота сопряжения
            file_bias = None
            metka_bias = 'found'
            for item in os.listdir(data_dir):
                if 'bias' in item and file_name in item and f'{file[indexes_h[0]-1]}km' in item:
                    file_bias = item
                    print(f' - {file} --> bias: {item}')
            
            if file_bias is None:
                if file == 'DC231008152952_2km.fits':
                    file_bias = 'DC230902202035_2km_bias.fits'
                    metka_bias = 'DC230902202035_2km_bias.fits'
                else:
                    if file.startswith('DC2202') or file.startswith('DC2203') or file.startswith('DC2205') or file.startswith('DC2211'):
                        file_bias = 'DC221108182951_2km_bias.fits'
                        metka_bias = 'DC221108182951_2km_bias.fits'
                        with fits.open(f'{data_dir}/{file}') as f:
                            print('file shape:', f.info())
                        with fits.open(f'{data_dir}/{file_bias}') as fb:
                            print('bias shape:', fb.info())
                print(f' - {file} --> bias: not found, took bias: {file_bias}')
                    
            with open('logs2.txt') as f:
                for line in f:
                    if file in line:
                        star_name = line.split()[1]
                        file_star = f'{line.split()[2].lower()}.sp'
                        alt = round(float(line.split()[-2]), 4)
                        az = round(float(line.split()[-1]), 4)
                        print(f' - logs.txt: {line.strip()} --> spectrum: {file_star}')
            
            processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=initial_params, dome_only=dome_only, use_windvar=use_windvar, star_name=star_name, latency_list=latency, alt=alt, az=az, do_crosscorr=do_crosscorr, metka_bias=metka_bias)
# ============================================================================

