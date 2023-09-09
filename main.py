import numpy as np
import matplotlib.pyplot as plt
import time

from reduction import processCorr # получение кросс-корр.
from tempfiles import processGamma # создание гамм
from initialparams import best_thresh, detect_peaks # начальные параметры для аппроксимации
from approx import processApprox # аппроксимация 

# 1. https://www.sthu.org/code/codesnippets/imagepers.html - туториал к опеределению положения пиков (вар. 2)
# 2. сдвиг изображения лучше реализовать через пиксели а не через скорости ветра, полученные параметры уже потом перевести
# в скорости ветра
# 3. можно ли заранее при создании гамм учесть домножение cjk? чтобы каждый раз не домножать внутри аппроксимации
# сjk ведь можно поделить на gamma_jk, чтобы в аппроксимации его не учитывать, ведь так? (как у матвея)

# чтение конфиг файла при запуске проги -------------------------------------------------------------------------
# with open('path.txt') as f:
#     for line in f:
#         data_dir = line

def processDomecam(file=None, file_bias=None, data_dir=None, D=None, conjugated_distance=None, latency=None, spectrum=None, lambda_=None, file_star=None, file_filter=None, file_ccd=None):
    # получение кросс-корр и автокорреляции зрачка 
    for lat in latency:
        cc, cjk, sec_per_frame = processCorr(file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
        # cc - картина кросс-корреляции
        # cjk - картина автокорреляции зрачка 
        # sec_per_frame - период между кадрами, [секунда]

        # создание гамм 
        gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd) 
    
        # подгонка начальных параметров для аппроксимации 
        thresh = best_thresh(cc, acc=5) # оптимальный трешхолд 
        y, x = detect_peaks(cc * (cc>thresh), size_of_neighborhood=7) # координаты целевых пиков

        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        ax.scatter(x, y, c='red', s=1)
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cc * (cc>thresh)), ax=ax2)
        fig.colorbar(ax3.imshow(cjk), ax=ax3)
        ax.set_title('Найденные пики')
        ax2.set_title('Оптимальный трешхолд')
        ax3.set_title('Автокорреляция зрачка')
        fig.suptitle('Вспомогательные картинки')
        
        # определение начальных параметров
        i_p = []
        all_Vy, all_Vx = [], []
        t = lat * sec_per_frame
        delta = D/(cc.shape[0]//2)
        a1 = np.linspace(0, 50000, 50)
        for i in range(len(x)):
            Vy = (cc.shape[0]//2-y[i])*delta/t
            Vx = -(cc.shape[1]//2-x[i])*delta/t
#             p0_Cn2_min = (cc[y[i], x[i]]/np.max(gammas[2km]))
#             p0_Cn2_max = (cc[y[i], x[i]]/np.max(gammas[50km]))
            i_p.append([Vy, Vx, 1, 10])
            all_Vy.append(Vy)
            all_Vx.append(Vx)
        
        fit = processApprox(cc=cc, gammas=gammas, lambda_=lambda_, D=D, latency=lat, sec_per_frame=sec_per_frame, gain=1, cjk=cjk, i_p=i_p, all_Vx=all_Vx, all_Vy=all_Vy, conjugated_distance=conjugated_distance)
    
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(fit), ax=ax2)
        fig.colorbar(ax3.imshow(cc-fit), ax=ax3)
        ax.set_title('orig')
        ax2.set_title('model')
        ax3.set_title('resid')
  