import time
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.filters import maximum_filter
from skimage.filters import threshold_multiotsu

def processBestThresh(img, acc=None):
    '''
    Функция для нахождения наилучшего значения threshold с помощью разных конфигураций алгоритма Оцу
    
    img - 2d изображение
    acc - макисмалньый процент полезного сигнала от всего изображения (по дефолту = 5%)
        это нужно, чтобы трешхолд не захватывал часть шума, который так же есть на изображении
    '''
    print('calculating threshold')
    st = time.perf_counter()
    img_size = img.shape[0]*img.shape[1]
    one_percent = img_size/100
    final_thresh = []
    
    for i in range(2, 5, 1):
        for j in range(len(threshold_multiotsu(img, classes=i))):
            thresh = threshold_multiotsu(img, classes=i)[j]
            found_values_size = len(np.where(img * (img > thresh) != 0)[1])
            found_percent = found_values_size/one_percent
            if thresh>0 and found_percent<acc:
                final_thresh.append(thresh)
    
    final_thresh = np.min(final_thresh)  
    
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - threshold: {final_thresh}')
    return final_thresh

def processPeakDetect(image, size_of_neighborhood=None):
    st = time.perf_counter()
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    
    size_of_neighborhood - размер области для поиска максимума
    """
    print('finding peaks')
    st = time.perf_counter()
    
    # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,2)
    neighborhood = np.ones((size_of_neighborhood, size_of_neighborhood))

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    
    y, x =np.where(detected_peaks != 0)
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - {len(y)} peaks found')
    return y, x

def processCoordsToSpeed(y, x, lat=None, sec_per_frame=None, D=None, cc=None):
    t = lat * sec_per_frame
    delta = D/(cc.shape[0]//2)
    all_Vy = np.zeros((len(x)), dtype=np.float32)
    all_Vx = np.zeros((len(x)), dtype=np.float32)
    for i in range(len(x)):
        all_Vy[i] = (cc.shape[0]//2-y[i])*delta/t
        all_Vx[i] = -(cc.shape[1]//2-x[i])*delta/t
    return all_Vy, all_Vx

def processCn2(cc, y, x, gammas, conjugated_distance=None):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if idx == (len(array) - 1):
            return idx, idx-1
        if idx == 0:
            return 1, 0
        else:
            if array[idx] > value:
                return idx, idx-1 
            if array[idx] < value:
                return idx+1, idx
            
    p0_Cn2 = np.zeros((len(x), 2), dtype=np.float32)
    
    num_of_layers=50
    a1 = np.linspace(0, 50000, num_of_layers)
    tmp = find_nearest(a1, conjugated_distance*1000)
    # возвращаются два индекса, я всегда беру наименьший из двух
    
    for i in range(len(x)):
        p0_Cn2_min = (cc[y[i], x[i]]/np.max(gammas[tmp[1]])) 
        p0_Cn2_max = (cc[y[i], x[i]]/np.max(gammas[num_of_layers-1]))
        p0_Cn2[i] = [p0_Cn2_min, p0_Cn2_max]
    return p0_Cn2, np.mean(p0_Cn2, axis=(1))