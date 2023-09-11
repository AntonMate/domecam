import time
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import correlate
from skimage.filters import threshold_otsu, gaussian
from astropy.io import fits

# ----------------------
def correlate1(frames, image_binary, latency): 
    print('cross correlating')
    st = time.perf_counter() 

# #   corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate()))) # np.real; np.abs
# #     correlation = [correlate(frames[i], frames[i + latency], mode='full', method='fft') 
# #                    for i in range(frames.shape[0] - latency)]
    
# #     iterable = (correlate(frames[i], frames[i + latency], mode='full', method='fft') 
# #                    for i in range(frames.shape[0] - latency))
# #     correlation = np.fromiter(iterable, dtype=np.dtype((np.float32, (2*frames.shape[1]-1, 2*frames.shape[2]-1))))
    
#     correlation = np.zeros((frames.shape[0] - latency, 2*frames.shape[1]-1, 2*frames.shape[2]-1), dtype=np.float32)
#     for i in range(frames.shape[0] - latency):
#         correlation[i] = correlate(frames[i], frames[i + latency], mode='full', method='fft')   

        
#     lol = np.mean(correlation) # такое ощущение, что из за этого следующий np.mean работает быстрее
#     res = np.mean(correlation, axis=0, dtype=np.float32)

#     res /= np.sum(image_binary, dtype=np.float32)

#     tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
#     tmp[1:,1:] = res
    
    tmp = np.loadtxt('test_cross_corr.gz')
    
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - cross-corr image shape: {tmp.shape[0]}x{tmp.shape[1]}')
    return tmp
    
# ----------------------
def processPupilWithCorr(images, latency): 
    print('data reduction')
    st = time.perf_counter()
    def image_square_cropp2(image): 
        mask = image != 0
        rows = np.flatnonzero((mask.any(axis=1))) 
        cols = np.flatnonzero((mask.any(axis=0)))
        res = image[rows.min():rows.max()+1, cols.min():cols.max()+1]
        return rows.min(), rows.max()+1, cols.min(), cols.max()+1
    
    def image_siz2(ar):
        if ar.shape[0] > ar.shape[1]:
            val = ar.shape[0] - ar.shape[1]
            ar = ar[:-val, :]
            return ar, val, 0
        if ar.shape[1] > ar.shape[0]:
            val = ar.shape[1] - ar.shape[0]
            ar = ar[:, :-val]
            return ar, 0, val
        if ar.shape[0] == ar.shape[1]:
            return ar, 0, 0
        
    def im_clean(images, mask):
        res = images * mask
        return res
    
    def im_norm(images, image_average):
        res = np.zeros_like(images, dtype=np.float32)
        for i in range(images.shape[0]):
            res[i] = ((images[i]/(image_average))*(np.sum(image_average)/np.sum(images[i])) - 1)
           
        return res
    
    image_average = np.mean(images, axis=0, dtype=np.float32) # средний кадр серии
    
    image_binary = (image_average > threshold_otsu(image_average)) # маска среднего кадра
    image_binary = np.array(image_binary, dtype=np.float32) # перевод в numpy
    y1, y2, x1, x2 = image_square_cropp2(image_binary)
    mask = image_binary[y1:y2, x1:x2]
    mask, yn, xn = image_siz2(mask)
    image_average=image_average[y1:y2-yn, x1:x2-xn]
    images = images[:, y1:y2-yn, x1:x2-xn]
    
    images_norm = im_norm(images, image_average) # нормировка изображений
    
    images_clean = im_clean(images_norm, mask) # отделение зрачка от фона
    images_clean[np.isnan(images_clean)] = 0

    res = images_clean[np.random.randint(images_clean.shape[0])]
    
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - pupil image shape: {res.shape[0]}x{res.shape[1]}')
    cross_corr = correlate1(images_clean, image_binary, latency)
    
    return res, cross_corr  
# ----------------------
def pupil(images, latency): 
    print('data reduction')
    st = time.perf_counter()
    def image_square_cropp(images): 
        mask = images[np.random.randint(images.shape[0])] != 0
        rows = np.flatnonzero((mask.any(axis=1))) 
        cols = np.flatnonzero((mask.any(axis=0)))
        res = images[:, rows.min():rows.max()+1, cols.min():cols.max()+1]
        return res    

    def image_size(ar):
        if ar.shape[1] > ar.shape[2]:
            ar = ar[:, :-(ar.shape[1] - ar.shape[2]), :]
            return ar
        if ar.shape[2] > ar.shape[1]:
            ar = ar[:, :, :-(ar.shape[2] - ar.shape[1])]
            return ar
        if ar.shape[1] == ar.shape[2]:
            return ar
    
        
    def im_clean(images, mask):
        res = images * mask
        return res
    
    def im_norm(images, image_average):
#         res = [(i/(image_average))*(np.sum(image_average)/np.sum(i)) - 1 for i in images]
        iterable = ((i/(image_average))*(np.sum(image_average)/np.sum(i)) - 1 for i in images)
        res = np.fromiter(iterable, dtype=np.dtype((np.float32, (images.shape[1], images.shape[2]))))
        return res
    
    st1 = time.perf_counter()
    image_average = np.mean(images, axis=0, dtype=np.float32) # средний кадр серии
    end1 = time.perf_counter()
    
    st2 = time.perf_counter()
    image_binary = (image_average > threshold_otsu(image_average)) # маска среднего кадра
    image_binary = np.array(image_binary, dtype=np.float32)
    end2 = time.perf_counter()
  
    st3 = time.perf_counter()
    images_norm = im_norm(images, image_average) # нормировка изображений
    end3 = time.perf_counter()
    
    st4 = time.perf_counter()
    images_clean = im_clean(images_norm, image_binary) # отделение зрачка от фона
    end4 = time.perf_counter()
        
    st5 = time.perf_counter()
    images_clean[np.isnan(images_clean)] = 0
    end5 = time.perf_counter()
    
    st6 = time.perf_counter()
    images_clean = image_square_cropp(images_clean) # обрезка зрачка по нулевым строкам и столбцам
    end6 = time.perf_counter()
    
    st7 = time.perf_counter()
    images_clean = image_size(images_clean) # подгонка размера изображений под квадратное
    end7 = time.perf_counter()
    
    st8 = time.perf_counter()
    res = images_clean[np.random.randint(images_clean.shape[0])]
    end8 = time.perf_counter()
    
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - pupil shape: {res.shape[0]}x{res.shape[1]}')
    cross_corr = correlate1(images_clean, image_binary, latency)
    
    return res, cross_corr  

def processAutoCorr(nx, frame):
    print('creating auto-corr pupil image')
    st = time.perf_counter()
    I0c = (frame != 0) * int(1)
    res = correlate(I0c, I0c, mode='full', method='fft')
    res = res / np.sum(frame!=0, dtype=np.float32)

    tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
    tmp[1:,1:] = res
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - auto-corr pupil image shape: {tmp.shape[0]}x{tmp.shape[1]}')
    return tmp

def processCorr(file=None, file_bias=None, D=None, latency=None, data_dir=None):
    print(f'{file}\n')
    print('collecting data')
    st = time.perf_counter() 
    with fits.open("".join([data_dir, '/', file])) as f:
        header = f[0].header
        sec_per_frame = 1/header['FRATE']
        data = np.float32(f[0].data)
        print(f' - time: {time.perf_counter() - st:.4f}')
        print(f' - data shape: {data.shape[0]}x{data.shape[1]}x{data.shape[2]}')
        
        if data.shape[1] > 246:
            print('WARNING: need binning')
        
        if file_bias is not None:
            print('collecting bias')
            st = time.perf_counter() 
            with fits.open("".join([data_dir, '/', file_bias])) as f:
                bias = np.mean(f[0].data, axis=0, dtype=np.float32)
                print(f' - time: {time.perf_counter() - st:.4f}')
                print(f' - bias image shape: {bias.shape[0]}x{bias.shape[1]}')
            data -= bias
        
        frame, data_corr = processPupilWithCorr(data, latency) # получение случайного изображения зрачка и изображения кросс-корреляции
        cjk = processAutoCorr(data_corr.shape[0], frame) # автокорреляция зрачка
        data_corr = gaussian(data_corr, sigma=1) # сглаживание изображения кросс-корреляции
    
#     print('Creating output image...')    
#     v = (D / data_corr.shape[0]) / (latency * sec_per_frame)
#     x = np.round(v*np.linspace(-data_corr.shape[0]//2+1, data_corr.shape[0]//2, 5), 2)
#     y = np.round(v*np.linspace(-data_corr.shape[0]//2+1, data_corr.shape[0]//2, 5), 2)
#     y = np.flipud(y)
   
#     plt.imshow(data_corr, cmap='gray')
#     plt.xticks(np.linspace(0, data_corr.shape[1], 5), labels=x)
#     plt.yticks(np.linspace(0, data_corr.shape[0], 5), labels=y)
#     plt.ylabel('Vy, m/s')
#     plt.xlabel('Vx, m/s')
#     plt.colorbar()
#     plt.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.2)
#     plt.savefig(f"{data_dir}/{file.replace('.fits', '')}.png", bbox_inches='tight')
#     print(f' - Done! Files saved to {data_dir}')
    return data_corr, cjk, sec_per_frame