import matplotlib.pyplot as plt
import time

from reduction import processCorr # получение кросс-корр.
from tempfiles import processGamma # создание гамм
from initialparams import best_thresh, detect_peaks # начальные параметры для аппроксимации
from approx import processApprox # аппроксимация 

# 1. подразумевается, что очищенное изображение зрачка квадратное и пока что оно по дефолту равно 226 на 226
# 2. https://www.sthu.org/code/codesnippets/imagepers.html - туториал к опеределению положения пиков (вар. 2)
# 3. сделать понятную реализацию для полихроматических гамм
# 4. сдвиг изображения лучше реализовать через пиксели а не через скорости ветра, полученные параметры уже потом перевести
# в скорости ветра
# 5. можно ли заранее при создании гамм учесть домножение cjk? чтобы каждый раз не домножать внутри аппроксимации
# сjk ведь можно поделить на gamma_jk, чтобы в аппроксимации его не учитывать, ведь так? (как у матвея)
# 6. период между кадрами считывать из фитс-файлов

# чтение конфиг файла при запуске проги -------------------------------------------------------------------------
# with open('path.txt') as f:
#     for line in f:
#         data_dir = line

def processDomecam(file=None, file_bias=None, data_dir=None, latency=None, spectrum=None, file_star=None, file_filter=None, file_ccd=None):
    

    # доп данные -------------------------------------------------------------------------
    D=2.5 # диаметр телескопа, [метр]
    lambda_=500*1e-9 # длина наблюдаемой волны света, [метр]

    # получение кросс-корр. ------------------------------------------------------------------------- 
    for lat in latency:
        cc, cjk, sec_per_frame = processCorr(file=file, file_bias=file_bias, D=D, latency=lat, data_dir=data_dir)
        # cc - картина кросс-корреляции
        # cjk - картина автокорреляции зрачка 
        # sec_per_frame - период между кадрами, [секунда]

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cjk), ax=ax2)
        ax.set_title('cross-corr image')
        ax2.set_title('pupil auto-corr image')

    # создание гамм -------------------------------------------------------------------------
        gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, sec_per_frame=sec_per_frame, latency=latency, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd) 
    
    # подгонка начальных параметров для аппроксимации -------------------------------------------------------------------------
        thresh = best_thresh(cc, acc=5) # порог макисмального значения трешхолда в КЛИН
        y, x = detect_peaks(cc * (cc>thresh), size_of_neighborhood=7) # координаты целевых пиков

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        ax.scatter(x, y, c='red', s=1)
        fig.colorbar(ax.imshow(cc), ax=ax)
        fig.colorbar(ax2.imshow(cc * (cc>thresh)), ax=ax2)
        ax.set_title('detected peaks')
        ax2.set_title('threshold')

        i_p = []
        t = lat * sec_per_frame
        delta = D/(cc.shape[0]//2)
        for i in range(len(x)):
    #         print('i.', y[i], x[i], cc[y[i], x[i]])
            Vy = (cc.shape[0]//2-y[i])*delta/t
            Vx = -(cc.shape[1]//2-x[i])*delta/t
            i_p.append([Vy, Vx, 1, 10])
        
#     fit = processApprox(cc=cc, gammas=gammas, D=D, latency=latency, sec_per_frame=sec_per_frame, gain=1, cjk=cjk, i_p=i_p)
    
#     fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5))
#     ax.scatter(x, y, c='red', s=1)
#     fig.colorbar(ax.imshow(cc), ax=ax)
#     fig.colorbar(ax2.imshow(fit), ax=ax2)
#     ax.set_title('detected peaks')
#     ax2.set_title('approx')
        
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