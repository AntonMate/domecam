import numpy as np
import matplotlib.pyplot as plt
import time
import os

from reduction import processCorr 
from tempfiles import processGamma 
from initialparams import processBestThresh, processPeakDetect, processCoordsToSpeed, processCn2
from approx import processApprox 

def processDomecam(file=None, file_bias=None, data_dir=None, D=None, conjugated_distance=None, latency=None, spectrum=None, lambda_=None, file_star=None, file_filter=None, file_ccd=None):
    # создание папки, где будут храниться изображения кросс-корр
    if not os.path.isdir("crosscorr"):
        os.mkdir("crosscorr")
        print('lol')
    for lat in latency:
        if os.path.isfile(f'crosscorr/{file[:-5]}_crosscorr_{lat}.gz'):
            cc, cjk, sec_per_frame = processCorr(run='no', file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
            cc = np.loadtxt(f'crosscorr/{file[:-5]}_crosscorr_{lat}.gz')
            # cjk = ?
            # sec_per_frame = ?
        else:
            # считывание данных, получение кросс-корр и автокорреляции зрачка 
            cc, cjk, sec_per_frame = processCorr(run='yes', file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
            # cc - картина кросс-корреляции
            # cjk - картина автокорреляции зрачка 
            # sec_per_frame - период между кадрами, [секунда]

        # создание теор. гамм 
        num_of_layers=50
        a1 = np.geomspace(100, 50000, num_of_layers)
        gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd, num_of_layers=num_of_layers, a1=a1) 
    
        # подсчет начальных параметров для аппроксимации 
        thresh = processBestThresh(cc, acc=5)
        y, x = processPeakDetect(cc * (cc>thresh), size_of_neighborhood=7)
        Vy, Vx = processCoordsToSpeed(y, x, lat=lat, sec_per_frame=sec_per_frame, D=D, cc=cc)
        p0_Cn2, p0_Cn2_mean = processCn2(cc/cjk, y, x, gammas, conjugated_distance=conjugated_distance, num_of_layers=num_of_layers, a1=a1)
        
        i_p = np.zeros((len(x), 4), dtype=np.float32)
        for i in range(len(x)):
            if int(Vx[i])==0 and int(Vy[i])==0:
                # тут можно для Cn2 брать значение Cn2 для conjugated_distance, а не Cn2_mean
                i_p[i] = [Vx[i], Vy[i], p0_Cn2[i][1], conjugated_distance]
            else:
                i_p[i] = [Vx[i], Vy[i], p0_Cn2_mean[i], 10]
        print('\nInitial guess for the parameters:')
        print(i_p)
        
        sum_cn2 = np.sum(p0_Cn2_mean*1e-13)        
        r0 = pow(0.423 * pow((2*np.pi/lambda_), 2) * sum_cn2, -3/5)
        seeing = 206265 * 0.98 * lambda_/r0
        print(' - total Cn2:', sum_cn2)
        print(f' - seeing, {lambda_/1e-9:.0f} nm: {seeing:.2f}')
        
        # thresh - оптимальный трешхолд картины кросс-корреляции 
        # y, x - координаты целевых пиков
        # Vy, Vx - скорости этих же пиков, [м/с]
        # p0_Cn2 - минимальное и максимальное значение интенсивонсти для каждого найденного пика
        # i_p - начальные параметры для каждого пика
        
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        ax.scatter(x, y, c='red', s=1)
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cc * (cc>thresh)), ax=ax2)
        fig.colorbar(ax3.imshow(cjk), ax=ax3)
        ax.set_title('Найденные пики')
        ax2.set_title('Оптимальный трешхолд')
        ax3.set_title('Автокорреляция зрачка')
        fig.suptitle('Вспомогательные картинки')
  
        fit = processApprox(cc=cc, gammas=gammas, lambda_=lambda_, D=D, latency=lat, sec_per_frame=sec_per_frame, cjk=cjk, i_p=i_p, all_Vx=Vx, all_Vy=Vy, p0_Cn2=p0_Cn2, conjugated_distance=conjugated_distance, num_of_layers=num_of_layers, a1=a1)
    
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(fit*cjk), ax=ax2)
        fig.colorbar(ax3.imshow(cc-(fit*cjk)), ax=ax3)
        ax.set_title('orig')
        ax2.set_title('model')
        ax3.set_title('resid')
  