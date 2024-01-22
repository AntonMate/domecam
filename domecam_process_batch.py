from main import processDomecam
import os
import getopt
import sys

# ============================================================================
latency = [4] # задержка для кросс-корреляции, [кадр]
conjugated_distance = 2 # сопряженное расстояние, [километр]
D = 2.5 # диаметр телескопа, [метр]
spectrum = 'poly' # тип излучения
lambda_ = 650*1e-9 # длина наблюдаемой волны света, [метр]
 
use_gradient = False # БС: использовать ли градиенты между слоями
use_windvar = True # использовать ли дисперсию ветра

# для полихроматического излучения нужны кривые фильтра, детектора и звезды
file_filter = 'KC19_d16t4_Safonov.xls'
file_ccd = 'ccd_prosilica2.crv'
# file_star = 'a05.sp'

# БС: если параметр do_fitting равен False: отладочный режим, аппроксимация не будет выполнена, а будут взяты начальные 
# параметры initial_params
# БС: если параметр do_fitting равен True, то будет выполнена оценка начальных параметров и проведена аппроксимация
do_fitting = True
dome_only = 5 # 0, чтобы отключить. >0, чтобы задать радиус области вокруг центра

# начальные параметры для аппроксимации в явном виде
# если их не указать, то они будут подобраны автоматически
# initial_params = [[   -0.03 ,    0.02, 2*0.5495173, 1.96746],
#                   [   -4.69 ,   -2.26, 5*1.019672,  3.23578],
#                   [   -6.38 ,   -6.27, 5*1.314183,  3.19194],
#                   [  -11.04 ,  -10.69, 2*1.060673,  9.83704],
#                   [  -17.26 ,  -16.99, 3*3.670290, 10.65737]]


# ============================================================================
optlist, args = getopt.getopt(sys.argv[1:], 'infile', ['infile='])
new_path = optlist[0][1]

if new_path.endswith('.fits'):
    print(f' - MODE: обработка одного файла {new_path}')
    indexes = [i for i in range(len(new_path)) if new_path[i] == "/"]
    data_dir = new_path[:indexes[-1]]
    print(' - dir:', data_dir)
    file = new_path[indexes[-1]+1:]
    print(' -', file)
    file_name = file.replace('_2km.fits', '')
    file_name = file_name.replace('_0km.fits', '') # номер серии

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
                print('test star:', star_name)
                file_star = f'{line.split()[2].lower()}.sp'
                print(f' - spectrum: {file_star}')
    
    processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=None, dome_only=dome_only, use_windvar=use_windvar)
                
else:
    print(f' - MODE: обработка всех файлов _2km.fits из {new_path}')
    data_dir = new_path
    for ser in os.listdir(new_path):
        if ser.endswith('_2km.fits'):
            print('')
            file = ser # номер серии
            print(' -', file)
            file_name = file.replace('_2km.fits', '') # номер серии
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
            
            processDomecam(file=file, file_name=file_name, file_bias=file_bias, data_dir=data_dir, D=D, conjugated_distance=conjugated_distance, latency=latency, spectrum=spectrum, lambda_=lambda_, file_filter=file_filter, file_ccd=file_ccd, file_star=file_star, do_fitting=do_fitting, use_gradient=use_gradient, initial_params=None, dome_only=dome_only, use_windvar=use_windvar, star_name=star_name)
# ============================================================================

