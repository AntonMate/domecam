import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.filters import threshold_multiotsu, threshold_otsu
from astropy.io import fits

def processApprox(cc=None, gammas=None, lambda_=None, D=None, latency=None, sec_per_frame=None, gain=None, cjk=None, i_p=None, all_Vx=None, all_Vy=None, conjugated_distance=None):
    k=50
    a1 = np.linspace(0, 50000, k)
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

        lv = find_nearest(a1, z)[1]
        uv = find_nearest(a1, z)[0]

        res = gammas[lv] + (z - a1[lv])*((gammas[uv] - gammas[lv])/(a1[uv] - a1[lv]))

        res = (res/(1e-13))*Cn2
        res = shift(res, (-Ypix, Xpix), order=1)  

        res = res * cjk
        return res

    def one_speckle_fit(params=None, data=None, lambda_=None, all_Vx=None, all_Vy=None, conjugated_distance=None): 
        def _g(M, *args): 
            x, y = M
            arr = np.zeros(x.shape)
            for i in range(len(args)//4):
                arr += gamma_se(x, y, *args[i*4:i*4+4]).ravel()
            return arr

        p0 = [p for prms in params for p in prms]
        print('approxing')
        st=time.perf_counter()

        x = np.linspace(-data.shape[1]//2, data.shape[1]//2-1, data.shape[1])
        y = np.linspace(-data.shape[0]//2, data.shape[0]//2-1, data.shape[0])
        X, Y = np.meshgrid(x, y)

        fit = np.zeros(X.shape) 
        xdata = np.vstack((X.ravel(), Y.ravel())) 
        ydata = data.ravel()
        
        lb, ub = [], []
        # в целом Cn2 можно тоже ограничить для каждого пика по диапазону высот: от 2км до 50км например
        for i in range(len(all_Vx)):
            lb.append([all_Vx[i]-0.5, all_Vy[i]-0.5, 0, conjugated_distance])
            ub.append([all_Vx[i]+0.5, all_Vy[i]+0.5, np.inf, 50])
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        lb_old = [-np.inf, -np.inf, 0, conjugated_distance]
        lb_old = np.tile(lb, len(p0)//4)
#         ub = [np.inf, np.inf, np.inf, 50]
#         ub = np.tile(ub, len(p0)//4)

        popt, pcov = curve_fit(_g, xdata, ydata, p0, bounds=[lb, ub])

        for i in range(len(popt)//4):
            fit += gamma_se(X, Y, *popt[i*4:i*4+4])

#     #     errors = np.sqrt(np.diag(pcov))

        popt = popt.reshape(len(popt)//4, 4)

        df = pd.DataFrame(popt, columns = ['Vx, m/s','Vy, m/s','Cn2, 1e-13', 'z, km'])
        df = df.sort_values(by=['z, km'])
        df = df.reset_index()
        df.drop(columns=['index'], inplace=True)

        sum_cn2 = np.sum(df['Cn2, 1e-13']*1e-13)        
        lambda_ = 500 * pow(10, -9)
        r0 = pow(0.423 * pow((2*np.pi/lambda_), 2) * sum_cn2, -3/5)
        seeing = 206265 * 0.98 * lambda_/r0
        
        print(f' - time: {time.perf_counter() - st:.4f}')
        print(' - total Cn2:', sum_cn2)
        print(f' - seeing, {lambda_/1e-9:.0f} nm: {seeing:.2f}')
        print(df.to_string(index=False))
        
        return fit
    
    fit = one_speckle_fit(params=i_p, data=cc, lambda_=lambda_, all_Vx=all_Vx, all_Vy=all_Vy, conjugated_distance=conjugated_distance)
    return fit
#         print()
#         return fit, popt
