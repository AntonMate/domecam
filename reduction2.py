import numpy as np
from astropy.io import fits
from skimage.filters import threshold_otsu, gaussian
from scipy.signal import correlate
import time


def processCorr(run_cc=None, file=None, bias=None, latencys=None, data_dir=None, dome_only=None, do_crosscorr=None):
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

    def image_clean(image, mask):
        res = image * mask
        return res

    def image_norm(image, image_average):
        res = np.zeros_like(image, dtype=np.float32)        
        res = ((image/(image_average))*(np.sum(image_average)/np.sum(image)) - 1)   
        return res

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

    def processAutoCorr(frame):
        st = time.perf_counter()
        I0c = (frame != 0) * int(1)
        res = correlate(I0c, I0c, mode='full', method='fft')
        res = res / np.sum(frame!=0, dtype=np.float32)
        tmp = np.zeros((res.shape[0]+1, res.shape[1]+1), dtype=np.float32)
        tmp[1:,1:] = res
        return tmp

    # =========================================================================================================
    # получение bias

    st = time.perf_counter()

    with fits.open(f'{data_dir}/{bias}') as f:
        if f[0].header['NAXIS'] == 3:
            bias = np.mean(np.float32(f[0].data), axis=0, dtype=np.float32)
        if f[0].header['NAXIS'] == 2:
            bias = np.float32(f[0].data)

    print(f' - time bias, {bias.shape}: {time.perf_counter() - st:.2f}')

    # =========================================================================================================
    # получение среднего кадра всей серии (и заодно периода между снимками)

    st = time.perf_counter()

    with fits.open(f'{data_dir}/{file}') as f:
        image_average =  np.mean(np.float32(f[0].data) - bias, axis=0, dtype=np.float32)

        header = f[0].header
        sec_per_frame = 1/header['FRATE']

    print(f' - time image_average: {time.perf_counter() - st:.2f}')

    # =========================================================================================================
    # обрезка среднего кадра, получение границ обрезки

    st = time.perf_counter()

    image_binary = (image_average > threshold_otsu(image_average)) # маска среднего кадра
    image_binary = np.array(image_binary, dtype=np.float32) # перевод в numpy
    y1, y2, x1, x2 = image_cropp(image_binary) # обрезка нулевых строк и столбцов 
    mask = image_binary[y1:y2, x1:x2]
    mask, yn, xn = image_resize(mask) # обрезка изображения под квадратное
    image_average=image_average[y1:y2-yn, x1:x2-xn]

    print(f' - time image_average_cropp: {time.perf_counter() - st:.2f}')

    # =========================================================================================================
    # по полученным границам далее обрезается каждый отдельный кадр серии и считается кросс корреляция

    if do_crosscorr:
        hdul = fits.open(f'{data_dir}/{file}')  
        nz = hdul[0].header['NAXIS3']
        corr_result = np.zeros((len(latencys), 2*((y2-yn)-y1), 2*((x2-xn)-x1)), dtype=np.float32)

        for k, latency in enumerate(latencys):
            st = time.perf_counter()

            correlation = np.zeros((nz-latency, 2*((y2-yn)-y1)-1, 2*((x2-xn)-x1)-1), dtype=np.float32)
            for i in range(0, nz-latency): # добавить потом nz-lantecy
                frame = hdul[0].section[i,:,:].astype(np.float32)
                frame -= bias
                frame = frame[y1:y2-yn, x1:x2-xn] # обрезка кадра 
                frame_norm = image_norm(frame, image_average) # нормировка кадра
                frame_clean = image_clean(frame_norm, mask) # отделение зрачка от фона
                frame_clean[np.isnan(frame_clean)] = 0 

                frame_latency = hdul[0].section[i+latency,:,:].astype(np.float32)
                frame_latency -= bias
                frame_latency = frame_latency[y1:y2-yn, x1:x2-xn] # обрезка кадра 
                frame_latency_norm = image_norm(frame_latency, image_average) # нормировка кадра
                frame_latency_clean = image_clean(frame_latency_norm, mask) # отделение зрачка от фона
                frame_latency_clean[np.isnan(frame_latency_clean)] = 0

                correlation[i] = correlate(frame_clean, frame_latency_clean, mode='full', method='fft')   

            res = np.mean(correlation, axis=0, dtype=np.float32)
            res /= np.sum(image_binary, dtype=np.float32)

            corr_result[k, 1:, 1:] = res
            corr_result[k] = gaussian(corr_result[k], sigma=1) # сглаживание изображения кросс-корреляции

            if dome_only != 0:
                corr_result[k] *= circle(dome_only, corr_result[k].shape[0], circle_centre=(0, 0), origin="middle")

        print(f' - time corr, latency {latency}: {time.perf_counter() - st:.2f}')
    if do_crosscorr == False:
        print(' - WARNING: no cross correlation count!')
        corr_result = np.zeros((len(latencys), 2*((y2-yn)-y1), 2*((x2-xn)-x1)), dtype=np.float32)

    # =========================================================================================================
    # подсчет автокорреляции зрачка, беру случайный кадр серии

    st = time.perf_counter()

    hdul = fits.open(f'{data_dir}/{file}')  

    frame_random = hdul[0].section[np.random.randint(hdul[0].header['NAXIS3']),:,:].astype(np.float32)
    frame_random -= bias
    frame_random = frame_random[y1:y2-yn, x1:x2-xn]
    frame_random_norm = image_norm(frame_random, image_average)
    frame_random_clean = image_clean(frame_random_norm, mask) 
    frame_random_clean[np.isnan(frame_random_clean)] = 0 
    cjk = processAutoCorr(frame_random_clean)

    print(f' - time autocorr: {time.perf_counter() - st:.2f}')
    
    return corr_result, cjk, sec_per_frame