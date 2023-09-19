import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.filters import threshold_multiotsu, threshold_otsu
from astropy.io import fits

def processApprox(cc=None, gammas=None, lambda_=None, D=None, latency=None, sec_per_frame=None, cjk=None, i_p=None, all_Vx=None, all_Vy=None, p0_Cn2=None, conjugated_distance=None, num_of_layers=None, a1=None):
    t = latency * sec_per_frame
    delta = D/(cc.shape[0]//2)
    
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

    def gamma_se(X, Y, Vx, Vy, Cn2, z): 
        Cn2=Cn2*1e-13
        z=z*1000

        Lx = Vx*t
        Ly = Vy*t
        Xpix = Lx/delta
        Ypix = Ly/delta

        uv, lv = find_nearest(a1, z) # lv - [1], ub - [0]
#         uv = find_nearest(a1, z)[0]
        res = gammas[lv] + (z - a1[lv])*((gammas[uv] - gammas[lv])/(a1[uv] - a1[lv]))
        
        res = shift(res, (-Ypix, Xpix), order=1)  
        return res

    def one_speckle_fit(params=None, data=None, lambda_=None, all_Vx=None, all_Vy=None, p0_Cn2=None, conjugated_distance=None): 
        def _g(M, *args): 
            x, y = M
            arr = np.zeros(x.shape)
            for i in range(len(args)//4):
                arr += gamma_se(x, y, *args[i*4:i*4+4]).ravel()
            return arr

        p0 = [p for prms in params for p in prms]
        print('approxing')
        st=time.perf_counter()

        x = np.linspace(-data.shape[1]//2, data.shape[1]//2-1, data.shape[1], dtype=np.float32)
        y = np.linspace(-data.shape[0]//2, data.shape[0]//2-1, data.shape[0], dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        fit = np.zeros(X.shape, dtype=np.float32) 
        xdata = np.vstack((X.ravel(), Y.ravel())) 
        ydata = data.ravel()
        
        # более точные баунсы для начальных параметров
        lb2 = np.zeros((len(p0)//4, 4), dtype=np.float32)
        ub2 = np.zeros((len(p0)//4, 4), dtype=np.float32)
        for i in range(len(all_Vx)):
            lb2[i] = [all_Vx[i]-0.5, all_Vy[i]-0.5, p0_Cn2[i][0]-0.005, conjugated_distance]
            ub2[i] = [all_Vx[i]+0.5, all_Vy[i]+0.5, p0_Cn2[i][1]+0.005, 50]
        lb2 = np.ravel(lb2)
        ub2 = np.ravel(ub2)
        
        lb = [-np.inf, -np.inf, 0, conjugated_distance]
        lb = np.tile(lb, len(p0)//4) 
        ub = [np.inf, np.inf, np.inf, 50]
        ub = np.tile(ub, len(p0)//4)
     
        popt, pcov = curve_fit(_g, xdata, ydata, p0, bounds=[lb2, ub2])

        for i in range(len(popt)//4):
            fit += gamma_se(X, Y, *popt[i*4:i*4+4])

#     #     errors = np.sqrt(np.diag(pcov))

        popt = popt.reshape(len(popt)//4, 4)

        df = pd.DataFrame(popt, columns = ['Vx, m/s','Vy, m/s','Cn2', 'z, m'])
        df = df.sort_values(by=['z, m'])
        df = df.reset_index()
        df['Cn2'] = df['Cn2']*1e-13
        df['z, m'] = df['z, m']*1000
        df = df.round({'Vx, m/s': 2})
        df = df.round({'Vy, m/s': 2})
        df = df.round({'z, m': 2})
        df.drop(columns=['index'], inplace=True)

        sum_cn2 = np.sum(df['Cn2'])        
        r0 = pow(0.423 * pow((2*np.pi/lambda_), 2) * sum_cn2, -3/5)
        seeing = 206265 * 0.98 * lambda_/r0
        
        print(f' - time: {time.perf_counter() - st:.4f}')
        print(df.to_string(index=False))
        print(' - total Cn2:', sum_cn2)
        print(f' - seeing, {lambda_/1e-9:.0f} nm: {seeing:.2f}')
        
        return fit
    
    # изображение кросс-корреляции делится на cjk, чтобы в аппроксимации не учитывать его домножение
    plt.figure()
    plt.imshow(cc)
    plt.title('cc')
    cc=cc/cjk
    cc[np.isinf(cc)] = 0
    plt.figure()
    plt.imshow(cc)
    plt.title('cc/cjk')
    fit = one_speckle_fit(params=i_p, data=cc, lambda_=lambda_, all_Vx=all_Vx, all_Vy=all_Vy, p0_Cn2=p0_Cn2, conjugated_distance=conjugated_distance)
    return fit
#         print()
#         return fit, popt
