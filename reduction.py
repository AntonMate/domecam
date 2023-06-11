import time
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import correlate
from skimage.filters import threshold_otsu, gaussian
from astropy.io import fits


# ----------------------
def correlate1(frames, image_binary, latency): 
    print('Cross correlating...')
    st = time.perf_counter() 
#   corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate()))) # np.real; np.abs
    correlation = [correlate(frames[i], frames[i + latency], mode='full', method='fft') 
                   for i in range(frames.shape[0] - latency)]
    
    
    res = np.mean(correlation, axis=0, dtype=np.float32)
    res /= np.sum(image_binary)
    
    tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
    tmp[1:,1:] = res
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - cross-correlation image shape: {tmp.shape[0]}x{tmp.shape[1]}')
    return tmp 
    
# ----------------------
def pupil(images, latency): 
    print('Data reduction...')
    st = time.perf_counter()
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
    
    res = images_clean[np.random.randint(images_clean.shape[0])]
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - pupil shape: {res.shape[0]}x{res.shape[1]}')
    cross_corr = correlate1(images_clean, image_binary, latency)
    
    return res, cross_corr  

def c_jk(nx, frame):
    I0c = (frame != 0) * int(1)
    res = correlate(I0c, I0c, mode='full', method='fft')
    res = res / np.sum(frame!=0)

    tmp = np.zeros((res.shape[0]+1, res.shape[1]+1))
    tmp[1:,1:] = res
    return tmp

def binning(image, factor=None):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    
    image = image[:-(image.shape[0] - factor*(image.shape[0]//factor)), :-(image.shape[1] - factor*(image.shape[1]//factor))]
    res = image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor)
    res = res.sum(1).sum(2)
    return res

def one(file=None, file_bias=None, bin_factor=None, D=None, latency=None, sec_per_frame=None, data_dir=None):
    print(f'{file}\n')
    print('Collecting data...')
    st = time.perf_counter() 
    with fits.open("".join([data_dir, '/', file])) as f:
        header = f[0].header
        data = np.float32(f[0].data)
        print(f' - Done! time: {time.perf_counter() - st:.4f}')
        print(f' - {data.shape[0]} pupil images shape: {data.shape[1]}x{data.shape[2]}')
        
        if data.shape[1] > 246:
            print('WARNING: need binning')
        
        if file_bias is not None:
            print('Collecting bias...')
            st = time.perf_counter() 
            with fits.open("".join([data_dir, '/', file_bias])) as f:
                bias = np.mean(f[0].data, axis=0, dtype=np.float32)
                print(f' - Done! time: {time.perf_counter() - st:.4f}')
                print(f' - bias shape: {bias.shape[0]}x{bias.shape[1]}')
        
        if bin_factor is not None:
            print('Binning...')
            data = np.array([binning(image, factor=bin_factor) for image in data])
            if file_bias is not None:
                bias = binning(bias, factor=bin_factor)
            print(f' - Done! New shape: {data.shape[1]}x{data.shape[2]}')

        if file_bias is not None:
            data -= bias
        
        frame, data_corr = pupil(data, latency)
        cjk = c_jk(data_corr.shape[0], frame)
        data_corr = gaussian(data_corr, sigma=1)
        if cjk.shape != data_corr.shape:
            print('WARNING: wrong cjk and corr shape')
    
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
    return data_corr, cjk