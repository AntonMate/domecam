import numpy as np
import matplotlib.pyplot as plt
import time
import os

from astropy.io import fits

from reduction2 import processCorr 
from tempfiles import processGamma 
from initialparams import processBestThresh, processPeakDetect, processCoordsToSpeed, processCn2
from approx import processApprox 
from checkfiles import processCheckFiles

def processDomecam(file=None, file_name=None, file_bias=None, data_dir=None, D=None, conjugated_distance=None, latency=None, spectrum=None, lambda_=None, file_star=None, file_filter=None, file_ccd=None, initial_params=None, use_gradient=None, do_fitting=None, dome_only=None, use_windvar=None):
    # считывание данных, получение кросс-корр и автокорреляции зрачка 
#     metka = processCheckFiles(file=file, latency=latency, data_dir=data_dir, dome_only=dome_only)
    
    
    metka = 'yes' # пока что убрал функцию подзагрузки старых файлов
    # создание папки, куда будут сохраняться результаты
    if not os.path.isdir(f'{data_dir}/results'):
        os.mkdir(f'{data_dir}/results')
    # создание папки с результатами обработки серии
    if not os.path.isdir(f'{data_dir}/results/{file_name}'):
        os.mkdir(f'{data_dir}/results/{file_name}')
    
    cc, cjk, sec_per_frame = processCorr(run_cc=metka, file=file, bias=file_bias, latencys=latency, data_dir=data_dir, dome_only=dome_only)
    
    # cc - картина кросс-корреляции
    # cjk - картина автокорреляции зрачка 
    # sec_per_frame - период между кадрами, [секунда]
    
    # создание теор. гамм 
    # 30 слоев до 30км работает быстрее, точность вроде бы та же
    st = time.perf_counter()
    
    num_of_layers=50
    heights_of_layers = np.geomspace(100, 50000, num_of_layers, dtype=np.float32)
    gammas = processGamma(lambda_, GammaType=spectrum, cjk=cjk, D=D, file=file, file_star=file_star, file_filter=file_filter, file_ccd=file_ccd, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, data_dir=data_dir, file_name=file_name) 
    
    print(f' - time creating {num_of_layers} {spectrum}chromatic turbulence layers, 0 to 50 km: {time.perf_counter() - st:.2f}')
    
    # подсчет начальных параметров для аппроксимации 
    # БС: начальные параметры будем определять по кросс-корряляции, посчитанной для минимальной задержки
    # БС: если начальные параметры были введены при вызове processDomecam, не оценивать их
    if do_fitting:
        for latency_i in [0]:
            st = time.perf_counter()
            thresh = processBestThresh(cc[latency_i], acc=5)
            y, x = processPeakDetect(cc[latency_i] * (cc[latency_i]>thresh), size_of_neighborhood=7)
            all_Vy, all_Vx = processCoordsToSpeed(y, x, latency=latency[latency_i], sec_per_frame=sec_per_frame, D=D, cc=cc[latency_i])
            with np.errstate(invalid='ignore'):
                all_Cn2_bounds, all_Cn2_mean = processCn2(cc[latency_i]/cjk, y, x, gammas, conjugated_distance=conjugated_distance, heights_of_layers=heights_of_layers)
            
            if use_windvar:
                initial_params = np.zeros((len(x), 5), dtype=np.float32)
                for i in range(len(x)):
                    if int(all_Vx[i])==0 and int(all_Vy[i])==0:
                        # тут можно для Cn2 брать значение Cn2 для conjugated_distance, а не Cn2_mean
                        initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_bounds[i][1], conjugated_distance, 2]
                        dome_index = i
                    else:
                        initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_mean[i], 10, 2]
            else:
                initial_params = np.zeros((len(x), 4), dtype=np.float32)
                for i in range(len(x)):
                    if int(all_Vx[i])==0 and int(all_Vy[i])==0:
                        # тут можно для Cn2 брать значение Cn2 для conjugated_distance, а не Cn2_mean
                        initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_bounds[i][1], conjugated_distance]
                        dome_index = i
                    else:
                        initial_params[i] = [all_Vx[i], all_Vy[i], all_Cn2_mean[i], 10]
            print(f' - time for initial params, {len(y)} peaks found: {time.perf_counter() - st:.2f}')
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
            
            xc=226
            if dome_only != 0:
                wind_speed_step = 51
                xs = 20
            if dome_only == 0:
                wind_speed_step = 5
                xs = 220
            
            xlims=(xc-xs,xc+xs)
            ylims=(xc+xs,xc-xs)
            
            v = (D / cjk.shape[0]) / (latency[latency_i] * sec_per_frame)
            x = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, wind_speed_step), 2)
            y = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, wind_speed_step), 2)
            y = np.flipud(y)
            ax.set_xticks(np.linspace(0, cjk.shape[1], wind_speed_step))
            ax.set_yticks(np.linspace(0, cjk.shape[0], wind_speed_step))
            ax.set_xticklabels(x)
            ax.set_yticklabels(y)
            ax.set_ylabel('Vy, m/s')
            ax.set_xlabel('Vx, m/s')
            
            ax2.set_xticks(np.linspace(0, cjk.shape[1], wind_speed_step))
            ax2.set_yticks(np.linspace(0, cjk.shape[0], wind_speed_step))
            ax2.set_xticklabels(x)
            ax2.set_yticklabels(y)
            ax2.set_ylabel('Vy, m/s')
            ax2.set_xlabel('Vx, m/s')
            
            x_cjk = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, 5), 2)
            y_cjk = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, 5), 2)
            y_cjk = np.flipud(y_cjk)
            ax3.set_xticks(np.linspace(0, cjk.shape[1], 5))
            ax3.set_yticks(np.linspace(0, cjk.shape[0], 5))
            ax3.set_xticklabels(x_cjk)
            ax3.set_yticklabels(y_cjk)
            ax3.set_ylabel('Vy, m/s')
            ax3.set_xlabel('Vx, m/s')

            ax2.set_xlim(xlims)
            ax2.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            
            plt.savefig(f'{data_dir}/results/{file_name}/{file[:-5]}_tmp.png')
            
    else:
        dome_index = 0
        all_Vx = None
        all_Vy = None
        all_Cn2_bounds = None
        all_Cn2_mean = None
        all_Cn2_bounds = None

    # БС: аппроксимацию выполняем по кросс-корреляциям, соответствующим всем задержкам одновременно        
    st=time.perf_counter()
    fit = processApprox(cc=cc, gammas=gammas, lambda_=lambda_, D=D, latency=latency, sec_per_frame=sec_per_frame, cjk=cjk, 
                        initial_params=initial_params, all_Vx=all_Vx, all_Vy=all_Vy, all_Cn2_bounds=all_Cn2_bounds, 
                        conjugated_distance=conjugated_distance, num_of_layers=num_of_layers, heights_of_layers=heights_of_layers, 
                        dome_index=dome_index, use_gradient=use_gradient, do_fitting=do_fitting, dome_only=dome_only, use_windvar=use_windvar, data_dir=data_dir, file=file, file_name=file_name)
    print(f' - time: {time.perf_counter() - st:.2f}')

    # БС: отображение результатов аппроксимации        
    xc=226
    if dome_only != 0:
        wind_speed_step = 51
        xs = 20
    if dome_only == 0:
        wind_speed_step = 5
        xs = 220
    xlims=(xc-xs,xc+xs)
    ylims=(xc+xs,xc-xs)
    vmin=-0.002
    vmax=0.008
    for latency_i in range(len(latency)):
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        fig.colorbar(ax.imshow(cc[latency_i], vmin=vmin, vmax=vmax), ax=ax)
        fig.colorbar(ax2.imshow(fit[latency_i]*cjk, vmin=vmin, vmax=vmax), ax=ax2)
        fig.colorbar(ax3.imshow((cc[latency_i]-fit[latency_i]*cjk), vmin=-0.004, vmax=0.004), ax=ax3)
        ax.set_title(f'orig, lat={latency[latency_i]}')
        v = (D / cjk.shape[0]) / (latency[latency_i] * sec_per_frame)
        x = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, wind_speed_step), 2)
        y = np.round(v*np.linspace(-cjk.shape[0]//2+1, cjk.shape[0]//2, wind_speed_step), 2)
        y = np.flipud(y)
        ax.set_xticks(np.linspace(0, cjk.shape[1], wind_speed_step))
        ax.set_yticks(np.linspace(0, cjk.shape[0], wind_speed_step))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)
        ax.set_ylabel('Vy, m/s')
        ax.set_xlabel('Vx, m/s')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        ax2.set_title(f'model, lat={latency[latency_i]}')
        ax2.set_xticks(np.linspace(0, cjk.shape[1], wind_speed_step))
        ax2.set_yticks(np.linspace(0, cjk.shape[0], wind_speed_step))
        ax2.set_xticklabels(x)
        ax2.set_yticklabels(y)
        ax2.set_ylabel('Vy, m/s')
        ax2.set_xlabel('Vx, m/s')
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        
        ax3.set_title(f'resid, lat={latency[latency_i]}')
        ax3.set_xticks(np.linspace(0, cjk.shape[1], wind_speed_step))
        ax3.set_yticks(np.linspace(0, cjk.shape[0], wind_speed_step))
        ax3.set_xticklabels(x)
        ax3.set_yticklabels(y)
        ax3.set_ylabel('Vy, m/s')
        ax3.set_xlabel('Vx, m/s')
        ax3.set_xlim(xlims)
        ax3.set_ylim(ylims)
        
        result = np.zeros((3, cjk.shape[0], cjk.shape[1]), dtype=np.float32)
        result[0] = cc[latency_i]
        result[1] = fit[latency_i]*cjk
        result[2] = cc[latency_i]-fit[latency_i]*cjk
        fits.writeto(f'{data_dir}/results/{file_name}/{file[:-5]}_{latency[latency_i]}_result.fits', result, overwrite=True)
        
    print(' - DONE!')