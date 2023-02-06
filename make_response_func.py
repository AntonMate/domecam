import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all="ignore")

def response_func():
    spectral_filter = pd.read_csv("KC19_d16t4_Safonov.csv", sep = ';')
    spectral_filter['14.42.43'] = spectral_filter['14.42.43'] / 100

    ccd_QE = pd.read_csv("ccd_prosilica2.crv", sep = ' ')
    ccd_QE['X'] = ccd_QE['X'] * 1000

    star = pd.read_csv("a05.sp", sep = ' ')

    # на данном этапе мы обрезаем значения файлов так, чтобы они покрывались всеми тремя файлами
    tmp_min = np.max([ccd_QE['X'][0], star['X'][0], spectral_filter['WAVE_LENGTH'][0]])
    tmp_max = np.min([ccd_QE['X'].iat[-1], star['X'].iat[-1], spectral_filter['WAVE_LENGTH'].iat[-1]])

    spectral_filter = spectral_filter[spectral_filter['WAVE_LENGTH'] < tmp_max]
    X1 = spectral_filter['WAVE_LENGTH'] * pow(10, -9) # [м]
    Y1 = spectral_filter['14.42.43']

    ccd_QE = ccd_QE[ccd_QE['X'] > tmp_min]
    X2 = ccd_QE['X'] * pow(10, -9) # [м]
    Y2 = ccd_QE['Y']

    star = star[star['X'] > tmp_min]
    X3 = star['X'] * pow(10, -9) # [м]
    Y3 = star['Y'] 

    # т.к. разбивка по осям отличается, нужно свести их к одной, потому что далее мы будем перемножать значения, чтобы
    # определить функцию спектрального отклика
    k = 1000
    max_lambda = int(tmp_max)
    lambdas = np.linspace(0, max_lambda, k) * pow(10, -9) # [м]

    interpolate_df = np.interp(lambdas, X1, Y1)
    interpolate_ccd = np.interp(lambdas, X2, Y2)
    interpolate_a05 = np.interp(lambdas, X3, Y3)
    result = interpolate_df * interpolate_ccd * interpolate_a05

    result = result/np.sum(result) # нормировка функции спектрального отклика
    lambdas = lambdas*pow(10, 9)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 5))

#     ax1.plot(X1, Y1, label = 'filter, r')
#     ax1.plot(X2, Y2, label = 'QE of ccd')
#     ax1.plot(X3, Y3, label = 'star, a05')
#     ax1.legend()
#     ax1.set_title('Data')
#     ax1.set_xlabel('λ, м')

#     ax2.plot(main_lambdas, main_result, c='black', label='F(λ)')
#     ax2.set_xlabel('λ, м')
#     ax2.legend()
#     ax2.set_title('Response function')
#     ax2.grid(color = 'black', linestyle='--', alpha = 0.2)
    return lambdas, result