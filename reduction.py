import time
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import correlate
from skimage.filters import threshold_otsu, gaussian
from astropy.io import fits

# ----------------------

def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    C = np.zeros((size, size), dtype=np.float32)
    coords = np.arange(0.5, size, 1.0, dtype=np.float32)
    if len(coords) != size:
        raise exceptions.Bug("len(coords) = {0}, ".format(len(coords)) +
                             "size = {0}. They must be equal.".format(size) +
                             "\n           Debug the line \"coords = ...\".")

    x, y = np.meshgrid(coords, coords)
    if origin == "middle":
        x -= size / 2.
        y -= size / 2.

    x -= circle_centre[0]
    y -= circle_centre[1]
    mask = x * x + y * y <= radius * radius
    C[mask] = 1
    return C


def correlate1(frames, image_binary, latency, dome_only=None): 
#   corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate()))) # np.real; np.abs
#     correlation = [correlate(frames[i], frames[i + latency], mode='full', method='fft') 
#                    for i in range(frames.shape[0] - latency)]
    
#     iterable = (correlate(frames[i], frames[i + latency], mode='full', method='fft') 
#                    for i in range(frames.shape[0] - latency))
#     correlation = np.fromiter(iterable, dtype=np.dtype((np.float32, (2*frames.shape[1]-1, 2*frames.shape[2]-1))))
    tmp = np.zeros((len(latency), 2*frames.shape[1], 2*frames.shape[2]), dtype=np.float32)
    for latency_i in range(len(latency)):
    
        correlation = np.zeros((frames.shape[0] - latency[latency_i], 2*frames.shape[1]-1, 2*frames.shape[2]-1), dtype=np.float32)
        for i in range(frames.shape[0] - latency[latency_i]):
            correlation[i] = correlate(frames[i], frames[i + latency[latency_i]], mode='full', method='fft')   
        
        res = np.mean(correlation, axis=0, dtype=np.float32)
        res /= np.sum(image_binary, dtype=np.float32)
        
        tmp[latency_i, 1:, 1:] = res
        tmp[latency_i] = gaussian(tmp[latency_i], sigma=1) # сглаживание изображения кросс-корреляции
        
        if dome_only != 0:
            tmp[latency_i] *= circle(dome_only, tmp[latency_i].shape[0], circle_centre=(0, 0), origin="middle")
    if len(latency) > 1:
            print(f' - latency {latency[latency_i]} done')
    
    return tmp
    
# ----------------------
def processPupilWithCorr(images, latency, dome_only=None): 
    print('data reduction')
    st = time.perf_counter()
    def image_cropp(image): 
        mask = image != 0
        rows = np.flatnonzero((mask.any(axis=1))) 
        cols = np.flatnonzero((mask.any(axis=0)))
        res = image[rows.min():rows.max()+1, cols.min():cols.max()+1]
        return rows.min(), rows.max()+1, cols.min(), cols.max()+1
    
    def image_resize(image):
        if image.shape[0] > image.shape[1]:
            val = image.shape[0] - image.shape[1]
            image = image[:-val, :]
            return image, val, 0
        if image.shape[1] > image.shape[0]:
            val = image.shape[1] - image.shape[0]
            image = image[:, :-val]
            return image, 0, val
        if image.shape[0] == image.shape[1]:
            return image, 0, 0
        
    def image_clean(images, mask):
        res = images * mask
        return res
    
    def image_norm(images, image_average):
        res = np.zeros_like(images, dtype=np.float32)
        for i in range(images.shape[0]):
            res[i] = ((images[i]/(image_average))*(np.sum(image_average)/np.sum(images[i])) - 1)
           
        return res
    
    image_average = np.mean(images, axis=0, dtype=np.float32) # средний кадр серии
    image_binary = (image_average > threshold_otsu(image_average)) # маска среднего кадра
    image_binary = np.array(image_binary, dtype=np.float32) # перевод в numpy
    y1, y2, x1, x2 = image_cropp(image_binary) # обрезка нулевых строк и столбцов 
    mask = image_binary[y1:y2, x1:x2]
    mask, yn, xn = image_resize(mask) # обрезка изображения под квадратное
    image_average=image_average[y1:y2-yn, x1:x2-xn]
    images = images[:, y1:y2-yn, x1:x2-xn]
    
    images_norm = image_norm(images, image_average) # нормировка изображений
    images_clean = image_clean(images_norm, mask) # отделение зрачка от фона
    images_clean[np.isnan(images_clean)] = 0

    random_pupil_image = images_clean[np.random.randint(images_clean.shape[0])]
    
    print(f' - pupil image shape: {random_pupil_image.shape}')
    print(f' - time: {time.perf_counter() - st:.2f}')
    
    print('cross correlating')
    st = time.perf_counter() 
    cc = correlate1(images_clean, image_binary, latency, dome_only=dome_only) # кросс-корреляция
    cjk = processAutoCorr(random_pupil_image) # автокорреляция зрачка
    print(f' - cross-corr image shape: {cc.shape}; auto-corr pupil image shape: {cjk.shape}')
    print(f' - time: {time.perf_counter() - st:.2f}')
    return random_pupil_image, cc, cjk
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
    
    image_average = np.mean(images, axis=0, dtype=np.float32) # средний кадр серии
    image_binary = (image_average > threshold_otsu(image_average)) # маска среднего кадра
    image_binary = np.array(image_binary, dtype=np.float32)
    images_norm = im_norm(images, image_average) # нормировка изображений
    images_clean = im_clean(images_norm, image_binary) # отделение зрачка от фона
    images_clean[np.isnan(images_clean)] = 0
    images_clean = image_square_cropp(images_clean) # обрезка зрачка по нулевым строкам и столбцам
    images_clean = image_size(images_clean) # подгонка размера изображений под квадратное
    res = images_clean[np.random.randint(images_clean.shape[0])]    
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - pupil shape: {res.shape[0]}x{res.shape[1]}')
    cross_corr = correlate1(images_clean, image_binary, latency)
    return res, cross_corr  

def processAutoCorr(frame):
    st = time.perf_counter()
    I0c = (frame != 0) * int(1)
    res = correlate(I0c, I0c, mode='full', method='fft')
    res = res / np.sum(frame!=0, dtype=np.float32)
    tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
    tmp[1:,1:] = res
    return tmp

def processCorr(run_cc=None, file=None, file_bias=None, D=None, latency=None, data_dir=None, dome_only=None):
    st = time.perf_counter() 
    if run_cc == 'yes':
        print('collecting data')
        with fits.open("".join([data_dir, '/', file])) as f:
            header = f[0].header
            sec_per_frame = 1/header['FRATE']
            data = np.float32(f[0].data)
            if file_bias is not None:
                with fits.open("".join([data_dir, '/', file_bias])) as f:
                    bias = np.mean(np.float32(f[0].data), axis=0, dtype=np.float32)
                print(f' - data shape: {data.shape}; bias shape: {bias.shape}')
                data -= bias
            else:
                print(f' - data shape: {data.shape}')
            print(f' - time: {time.perf_counter() - st:.2f}')
            
            frame, data_corr, cjk = processPupilWithCorr(data, latency, dome_only=dome_only)

            for latency_i in range(len(latency)):
                if dome_only != 0:
                    np.save(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{latency[latency_i]}_dome.npy', data_corr[latency_i])
                if dome_only == 0:
                    np.save(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{latency[latency_i]}.npy', data_corr[latency_i])
            np.save(f'{data_dir}/crosscorr/{file[:-5]}_cjk.npy', cjk)
    
    if run_cc == 'no':
        print('WARNING: cross correlation is loaded from old file')
        cjk = np.load(f'{data_dir}/crosscorr/{file[:-5]}_cjk.npy')
        data_corr = np.zeros((len(latency), cjk.shape[0], cjk.shape[1]), dtype=np.float32)
        for latency_i in range(len(latency)):
            if dome_only != 0:
                data_corr[latency_i] = np.load(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{latency[latency_i]}_dome.npy')    
            if dome_only == 0:
                data_corr[latency_i] = np.load(f'{data_dir}/crosscorr/{file[:-5]}_crosscorr_{latency[latency_i]}.npy')    
        
        with fits.open("".join([data_dir, '/', file])) as f:
            header = f[0].header
            sec_per_frame = 1/header['FRATE']
           
    return data_corr, cjk, sec_per_frame