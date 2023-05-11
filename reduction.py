import time
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import correlate
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from astropy.io import fits

# ----------------------
def correlate1(frames, image_binary, latency): 
#   corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate()))) # np.real; np.abs
    correlation = [correlate(frames[i], frames[i + latency], mode='full', method='fft') 
                   for i in range(frames.shape[0] - latency)]
    res = np.mean(correlation, axis=0, dtype=np.float32)
    res /= np.sum(image_binary)
    
    tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
    tmp[1:,1:] = res
    return tmp 
    
# ----------------------
def pupil(images, latency): 
    def image_square_cropp(images): 
        mask = images[np.random.randint(images.shape[0])] != 0
        rows = np.flatnonzero((mask.any(axis=1))) 
        cols = np.flatnonzero((mask.any(axis=0)))
        res = images[:, rows.min():rows.max()+1, cols.min():cols.max()+1]
        return res    
    
    def image_size(image):
        if image.shape[1] != 226 or image.shape[2] != 226:
            tmp = np.zeros((image.shape[0], image.shape[1] - (image.shape[1] - 226), image.shape[2] - (image.shape[2] - 226)), 
                           dtype=np.float32)
            image = image[:, 0:image.shape[1] - (image.shape[1] - 226), 0:image.shape[2] - (image.shape[2] - 226)]
            if image.shape[1] < 226 and image.shape[2] == 226:
                tmp[:, (226-image.shape[1]):, :] = image
                return tmp
            if image.shape[2] < 226 and image.shape[1] == 226:
                tmp[:, :, (226-image.shape[2]):] = image
                return tmp
            else:
                return image
        else:
            return image
    
    image_average = np.mean(images, axis=0) # средний кадр серии
    
    image_binary = (image_average > threshold_otsu(image_average)*int(1)) # маска среднего кадра
    
    images_norm = [(i/(image_average))*(np.sum(image_average)/np.sum(i)) - 1 for i in images] # нормировка изображений
    images_clean = images_norm * image_binary # отделение зрачка от фона
    images_clean[np.isnan(images_clean)] = 0
    
    images_clean = image_square_cropp(images_clean) # обрезка зрачка в квадрат
    images_clean = image_size(images_clean) # подгонка размера изображений зрачка под 226х226
    
    cross_corr = correlate1(images_clean, image_binary, latency)
    res = images_clean[np.random.randint(images_clean.shape[0])]
    return res, cross_corr  

def binning(image, factor=None):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    
    image = image[:-(image.shape[0] - factor*(image.shape[0]//factor)), :-(image.shape[1] - factor*(image.shape[1]//factor))]
    res = image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor)
    res = res.sum(1).sum(2)
    return res

def start(file=None, file_bias=None, bin_factor=None, D=None, latency=None, data_dir=None, save_as=None):
    st = time.perf_counter() 
   
    with fits.open("".join([data_dir, '/', file])) as f:
        f.info()
        header = f[0].header
        sec_per_frame = 0.01
        data = np.float32(f[0].data)
    
        if bin_factor is not None:
            data = np.array([binning(image, factor=bin_factor) for image in data])
            print('binning done:', data.shape)
    
        if file_bias is not None:
            with fits.open("".join([data_dir, '/', file_bias])) as f:
                f.info()
                bias = np.mean(f[0].data, axis=0, dtype=np.float32)
        
            data -= bias
    
        frame, data_corr = pupil(data, latency)
    
        print('cross corr latency:', latency)
        print('pupil image shape:', frame.shape)
        print('cross corr image shape:', data_corr.shape)
        print('max cross corr value:', np.unravel_index(np.argmax(data_corr), data_corr.shape), np.max(data_corr))
        print('min cross corr value:', np.unravel_index(np.argmin(data_corr), data_corr.shape), np.min(data_corr))
    
    np.savetxt(f'{data_dir}/{save_as}.gz', data_corr)
    np.savetxt(f'{data_dir}/pupil.gz', frame)
    np.savetxt(f'{data_dir}/{save_as}_blur.gz', gaussian(data_corr, sigma=1))
    print('\nfiles saved!')
    print('time:', time.perf_counter()-st)
    
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5)) 
    ax.imshow(frame, cmap='gray')
    ax2.imshow(data_corr, cmap='gray')
    ax.set_title('random pupil image')
    ax2.set_title('cross-corr')
    
    if D is not None:
        v = (D / data_corr.shape[1]) / (latency * sec_per_frame)
        x = np.round(v*np.linspace(-data_corr.shape[0]//2+1, data_corr.shape[0]//2, 5), 2)
        y = np.round(v*np.linspace(-data_corr.shape[0]//2+1, data_corr.shape[0]//2, 5), 2)
        y = np.flipud(y)

        ax2.set_xticks(np.linspace(0, data_corr.shape[1], 5))
        ax2.set_yticks(np.linspace(0, data_corr.shape[0], 5))
        ax2.set_xticklabels(x)
        ax2.set_yticklabels(y)
        ax2.set_ylabel('Vy, m/s')
        ax2.set_xlabel('Vx, m/s')
    
    ax2.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
    plt.show()