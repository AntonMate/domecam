import os
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
plt.style.use(astropy_mpl_style)
# kadr_num  = 0 # номер кадра для отображения на графиках

# трешхолд по Отцу для бинаризации
def otsu_tresh(gray):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(gray, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    np.seterr(invalid='ignore')
    for t in bins[1:-1]: 
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])   
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight
        mub = np.sum(intensity_arr[:t]*his[:t]) / np.float32(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / np.float32(pcf)
        value = Wb * Wf * (mub - muf) ** 2
        if value > final_value:
            final_thresh = t
            final_value = value
    return final_thresh

# средний кадр серии
def avr_data(data):
    return np.mean(data, axis=0, dtype='float32')

# нормировка кадров серии
def norm_data(data):
    return np.float32((data)/(avr_data(data)) - 1)
 
# бинаризация кадров
def im_bin(data):
    return np.float32((avr_data(data) > otsu_tresh(avr_data(data))) * int(255))

# вырезка зрачка
def otsu_res(data):
    return np.float32(norm_data(data) * im_bin(data))

# обрезка зрачка в квадрат
def sq_cropp(data):
    mask = otsu_res(data)[0] != 0
    rows = np.flatnonzero((mask.any(axis=1))) 
    cols = np.flatnonzero((mask.any(axis=0)))
    squared = otsu_res(data)[:, rows.min():rows.max()+1, cols.min():cols.max()+1] 
    print(' ')
    print('size of cropped image:', squared.shape)
    return squared 

# рассчет скорости по осям 
def v(D, img, latency, frames_per_sec):
    k = D / img.shape[2]
    print('1 pixel equals to', k, 'meters')
    return (D / img.shape[2]) / (latency * frames_per_sec)

def cross_corr_ft(img1, img2):
    corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate())))
    corr /= np.max(corr)
    return corr

def cross_corr(img, D, latency, frames_per_sec):
    all_corr = np.zeros((img.shape))
    for i in range(0, img.shape[0]-latency, 1):
        all_corr[i] = cross_corr_ft(img[i], img[i+latency])
    avr_cross = np.mean(all_corr, axis=0, dtype='float32')
    print('size of cross_corr image:', avr_cross.shape)
    
    x = np.round(v(D, img, latency, frames_per_sec)*np.linspace(-avr_cross.shape[0]//2, avr_cross.shape[0]//2, 7), 2)
    fig = plt.figure()
    ax = plt.axes()
    im = plt.imshow(avr_cross)
    ax.set_xticks(np.linspace(0, avr_cross.shape[0], 7))
    ax.set_yticks(np.linspace(0, avr_cross.shape[0], 7))
    ax.set_xticklabels(x, fontsize=12)
    ax.set_yticklabels(x, fontsize=12)
    ax.set_ylabel('Vy, m/s', fontsize=12)
    ax.set_xlabel('Vx, m/s', fontsize=12)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    ax.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
    # plt.savefig(f'C:/astro/corr_with_latency_{latency}.png', bbox_inches='tight')
    
    
def domecam(file, file_bias, D, latency):
    st = time.perf_counter()
    with fits.open(os.path.abspath(file_bias)) as df:
        df.info()
        avr_bias = np.mean(df[0].data, axis=0, dtype='float32')      
    with fits.open(os.path.abspath(file)) as df: 
        df.info()
        frames_per_sec = 1 / df[0].header['FRATE']
        cross_corr(sq_cropp(df[0].data - avr_bias), D, latency, frames_per_sec)   
    print(' ')
    print('time: ', time.perf_counter() - st)
    gc.collect()
        