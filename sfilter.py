import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/miron/Downloads/KC19_d16t4_Safonov.csv", sep = ';')
df['14.42.43'] = df['14.42.43'] / 100

ccd = pd.read_csv("C:/Users/miron/Downloads/ccd_prosilica2.crv", sep = ' ')
ccd['X'] = ccd['X'] * 1000

a05 = pd.read_csv("C:/Users/miron/Downloads/a05.sp", sep = ' ')

def sfilter(df, ccd, a05, D=2.5, D_pix=228, z=2000):
    # на данном этапе мы обрезаем значения файлов так, чтобы они покрывались всеми тремя файлами
    tmp_min = np.max([ccd['X'][0], a05['X'][0], df['WAVE_LENGTH'][0]])
    tmp_max = np.min([ccd['X'].iat[-1], a05['X'].iat[-1], df['WAVE_LENGTH'].iat[-1]])

    df = df[df['WAVE_LENGTH'] < tmp_max]
    X1 = df['WAVE_LENGTH'] * pow(10, -9) # [м]
    Y1 = df['14.42.43']

    ccd = ccd[ccd['X'] > tmp_min]
    X2 = ccd['X'] * pow(10, -9) # [м]
    Y2 = ccd['Y']

    a05 = a05[a05['X'] > tmp_min]
    X3 = a05['X'] * pow(10, -9) # [м]
    Y3 = a05['Y'] 

    coeff = 1
    k = 1000*coeff
    max_lambda = tmp_max*coeff
    lambdas = np.linspace(0, max_lambda, k) * pow(10, -9) # [м]

    interpolate_df = np.interp(lambdas, X1, Y1)
    interpolate_ccd = np.interp(lambdas, X2, Y2)
    interpolate_a05 = np.interp(lambdas, X3, Y3)
    result = interpolate_df * interpolate_ccd * interpolate_a05
    result = result/np.sum(result)

    # мнимая часть Фурье преобразования от функции спектрального отклика
    delta_lambdas = (max_lambda / len(lambdas)) * pow(10, -9) # период дискредизации, [м]
    omega_lambdas_scale = 1 / (delta_lambdas) # максималльноешаг по частоте, [м^-1]
    res_fft = pow((np.imag(np.fft.fft(result/lambdas))), 2)

    nx = 512 # размер окна, [n_pix]

    delta = D/D_pix # шаг субапертуры, период дискретизации (то, насколько одно значение отстает от следующего) [м]
    
    f_scale = 1/(delta*nx) # шаг по частоте, [м^-1]
    xx, yy = np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx))
    xx_scale = f_scale * xx 
    yy_scale = f_scale * yy
    f_abs = np.sqrt(pow(xx_scale, 2) + pow(yy_scale, 2))

    omega = 0.5 * z * pow(f_abs, 2) # аргумент, шаг по частоте, [м^-1]
    omega = np.ravel(omega)
    omega_new = np.interp(omega, np.linspace(0, omega_lambdas_scale, k), res_fft)
    omega_new = np.resize(omega_new, (nx, nx))
    
    return omega_new   