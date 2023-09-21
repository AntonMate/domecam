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
    if not os.path.isdir(f"{data_dir}/crosscorr"):
        os.mkdir(f"{data_dir}/crosscorr")
    for lat in latency:
        if os.path.isfile(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{lat}.npy'):
            cc, cjk, sec_per_frame = processCorr(run_cc='no', file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
        else:
            # считывание данных, получение кросс-корр и автокорреляции зрачка 
            cc, cjk, sec_per_frame = processCorr(run_cc='yes', file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
            # cc - картина кросс-корреляции
            # cjk - картина автокорреляции зрачка 
            # sec_per_frame - период между кадрами, [секунда]

        # создание теор. гамм 
        # конфигурация на 30 слоев до 30км работает быстрее, точность вроде бы та же
        num_of_layers=50
        heights_of_layers = np.geomspace(100, 50000, num_of_layers, dtype=np.float32)
        gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, data_dir=data_dir) 
    
        # подсчет начальных параметров для аппроксимации 
        thresh = processBestThresh(cc, acc=5)
        y, x = processPeakDetect(cc * (cc>thresh), size_of_neighborhood=7)
        all_Vy, all_Vx = processCoordsToSpeed(y, x, lat=lat, sec_per_frame=sec_per_frame, D=D, cc=cc)
        all_Cn2_bounds, all_Cn2_mean = processCn2(cc/cjk, y, x, gammas, conjugated_distance=conjugated_distance, heights_of_layers=heights_of_layers)
        
        initial_params = np.zeros((len(x), 4), dtype=np.float32)
        for i in range(len(x)):
            if int(all_Vx[i])==0 and int(all_Vy[i])==0:
                # тут можно для Cn2 брать значение Cn2 для conjugated_distance, а не Cn2_mean
                initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_bounds[i][1], conjugated_distance]
                dome_index = i
            else:
                initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_mean[i], 10]
      
        # thresh - оптимальный трешхолд картины кросс-корреляции 
        # y, x - координаты целевых пиков
        # Vy, Vx - скорости этих же пиков, [м/с]
        # p0_Cn2 - минимальное и максимальное значение интенсивонсти для каждого найденного пика
        # i_p - начальные параметры для каждого пика
        
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        ax.scatter(x, y, c='red', marker='x', s=1)
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cc * (cc>thresh)), ax=ax2)
        fig.colorbar(ax3.imshow(cjk), ax=ax3)
        ax.set_title('Найденные пики')
        ax2.set_title('Оптимальный трешхолд')
        ax3.set_title('Автокорреляция зрачка')
        fig.suptitle('Вспомогательные картинки')
  
        fit = processApprox(cc=cc, gammas=gammas, lambda_=lambda_, D=D, latency=lat, sec_per_frame=sec_per_frame, cjk=cjk, initial_params=initial_params, all_Vx=all_Vx, all_Vy=all_Vy, all_Cn2_bounds=all_Cn2_bounds, conjugated_distance=conjugated_distance, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, dome_index=dome_index)
    
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(fit*cjk), ax=ax2)
        fig.colorbar(ax3.imshow(cc-(fit*cjk)), ax=ax3)
        ax.set_title('orig')
        ax2.set_title('model')
        ax3.set_title('resid')
  