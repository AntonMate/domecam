import numpy as np
import matplotlib.pyplot as plt
import time

from reduction import processCorr # получение кросс-корр.
from tempfiles import processGamma # создание гамм
from initialparams import best_thresh, detect_peaks # начальные параметры для аппроксимации
from approx import processApprox # аппроксимация 

# 2. https://www.sthu.org/code/codesnippets/imagepers.html - туториал к опеределению положения пиков (вар. 2)
# 4. сдвиг изображения лучше реализовать через пиксели а не через скорости ветра, полученные параметры уже потом перевести
# в скорости ветра
# 5. можно ли заранее при создании гамм учесть домножение cjk? чтобы каждый раз не домножать внутри аппроксимации
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

#         v = (D / cjk.shape[0]/2) / (lat * sec_per_frame)
#         x = np.round(v*np.linspace(-cc.shape[0]//2+1, cc.shape[0]//2, 5), 2)
#         y = np.round(v*np.linspace(-cc.shape[0]//2+1, cc.shape[0]//2, 5), 2)
#         y = np.flipud(y)
#         fig = plt.figure()
#         ax = plt.axes()
#         im = plt.imshow(cc)
#         ax.set_xticks(np.linspace(0, cc.shape[1], 5))
#         ax.set_yticks(np.linspace(0, cc.shape[0], 5))
#         ax.set_xticklabels(x)
#         ax.set_yticklabels(y)
#         ax.set_ylabel('Vy, m/s')
#         ax.set_xlabel('Vx, m/s')
#         cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
#         plt.colorbar(im, cax=cax)
#         ax.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)

        # создание гамм 
        gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, sec_per_frame=sec_per_frame, latency=latency, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd) 
    
        # подгонка начальных параметров для аппроксимации 
        thresh = best_thresh(cc, acc=5) # оптимальный трешхолд для КЛИН
        y, x = detect_peaks(cc * (cc>thresh), size_of_neighborhood=7) # координаты целевых пиков

        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        ax.scatter(x, y, c='red', s=1)
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cc * (cc>thresh)), ax=ax2)
        fig.colorbar(ax3.imshow(cjk), ax=ax3)
        ax.set_title('Найденные пики')
        ax2.set_title('Оптимальный трешхолд для КЛИН')
        ax3.set_title('Автокорреляция зрачка')
        fig.suptitle('Вспомогательные картинки')

        i_p = []
        all_Vy, all_Vx = [], []
#         p0_Cn2 = (res[my, mx]/np.max(gamma_poly_se(X, Y, Vx, Vy, 10, 2))) * 10
        t = lat * sec_per_frame
        delta = D/(cc.shape[0]//2)
        for i in range(len(x)):
            Vy = (cc.shape[0]//2-y[i])*delta/t
            Vx = -(cc.shape[1]//2-x[i])*delta/t
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
        
    # аппроксимация 
    
    # from clean import curvef

# curvef(file=f'data/20210130j_m2km_corr_4_blur.gz', #c,d,e,g,h,i,j,l,m
#        file2=None,
#        mode='blur', 
#        D=2.5, 
#        latency=4, 
#        sec_per_frame=0.01, 
#        dist0=2, # нужно для отрисовки профиля, отнимается от file
#        dist02=0,
#        gain=1, 
#        thresh_manual=0, 
#        thresh_manual2=0,
#        niter=50, 
#        window=15,
#        run_clean='yes',
#        checkpointf='no', 
#        step=None,
#        seeing_lambda=500*pow(10, -9),
#        data_dir='D:/astro/domecam')