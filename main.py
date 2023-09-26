import numpy as np
import matplotlib.pyplot as plt
import time
import os

from reduction import processCorr 
from tempfiles import processGamma 
from initialparams import processBestThresh, processPeakDetect, processCoordsToSpeed, processCn2
from approx import processApprox 
from checkfiles import processCheckFiles

def processDomecam(file=None, file_bias=None, data_dir=None, D=None, conjugated_distance=None, latency=None, spectrum=None, lambda_=None, file_star=None, file_filter=None, file_ccd=None):
    # считывание данных, получение кросс-корр и автокорреляции зрачка 
    metka = processCheckFiles(file=file, latency=latency, data_dir=data_dir)
    cc, cjk, sec_per_frame = processCorr(run_cc=metka, file=file, file_bias=file_bias, D=D, latency=latency, data_dir=data_dir)
    # cc - картина кросс-корреляции
    # cjk - картина автокорреляции зрачка 
    # sec_per_frame - период между кадрами, [секунда]
    
    # создание теор. гамм 
    # 30 слоев до 30км работает быстрее, точность вроде бы та же
    print('creating gammas')
    st = time.perf_counter()
    
    num_of_layers=50
    heights_of_layers = np.geomspace(100, 50000, num_of_layers, dtype=np.float32)
    gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, data_dir=data_dir) 
    print(f' - {num_of_layers} {spectrum}chromatic turbulence layers from 0 to 50 km')
    print(f' - time: {time.perf_counter() - st:.2f}')
    
    # подсчет начальных параметров для аппроксимации 
    for latency_i in range(len(latency)):
        print('calculation initial parametrs')
        st = time.perf_counter()
        thresh = processBestThresh(cc[latency_i], acc=5)
        y, x = processPeakDetect(cc[latency_i] * (cc[latency_i]>thresh), size_of_neighborhood=7)
        all_Vy, all_Vx = processCoordsToSpeed(y, x, latency=latency[latency_i], sec_per_frame=sec_per_frame, D=D, cc=cc[latency_i])
        all_Cn2_bounds, all_Cn2_mean = processCn2(cc[latency_i]/cjk, y, x, gammas, conjugated_distance=conjugated_distance, heights_of_layers=heights_of_layers)
        
        initial_params = np.zeros((len(x), 4), dtype=np.float32)
        for i in range(len(x)):
            if int(all_Vx[i])==0 and int(all_Vy[i])==0:
                # тут можно для Cn2 брать значение Cn2 для conjugated_distance, а не Cn2_mean
                initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_bounds[i][1], conjugated_distance]
                dome_index = i
            else:
                initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_mean[i], 10]
        print(f' - threshold: {thresh:.4f}; {len(y)} peaks found')
        print(f' - time: {time.perf_counter() - st:.2f}')
        # thresh - оптимальный трешхолд картины кросс-корреляции 
        # y, x - координаты пиков
        # all_Vy, all_Vx - скорости этих же пиков, [м/с]
        # all_Cn2_bounds - минимальное и максимальное значение интенсивонсти для каждого найденного пика
        # initial_params - начальные параметры для каждого пика
        
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        ax.scatter(x, y, c='red', marker='x', s=1)
        fig.colorbar(ax.imshow(cc[latency_i]), ax=ax)
        fig.colorbar(ax2.imshow(cc[latency_i] * (cc[latency_i]>thresh)), ax=ax2)
        fig.colorbar(ax3.imshow(cjk), ax=ax3)
        ax.set_title('Найденные пики')
        ax2.set_title('Оптимальный трешхолд')
        ax3.set_title('Автокорреляция зрачка')
        fig.suptitle('Вспомогательные картинки')
        
        print('approxing')
        st=time.perf_counter()
        fit = processApprox(cc=cc[latency_i], gammas=gammas, lambda_=lambda_, D=D, latency=latency[latency_i], sec_per_frame=sec_per_frame, cjk=cjk, initial_params=initial_params, all_Vx=all_Vx, all_Vy=all_Vy, all_Cn2_bounds=all_Cn2_bounds, conjugated_distance=conjugated_distance, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, dome_index=dome_index)
        print(f' - time: {time.perf_counter() - st:.2f}')
        
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        fig.colorbar(ax.imshow(cc[latency_i]), ax=ax)
        fig.colorbar(ax2.imshow(fit*cjk), ax=ax2)
        fig.colorbar(ax3.imshow(cc[latency_i]-(fit*cjk)), ax=ax3)
        ax.set_title('orig')
        ax2.set_title('model')
        ax3.set_title('resid')
  