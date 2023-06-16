import time
import numpy as np
from skimage.filters import gaussian

np.seterr(divide='ignore')

# --- апертурный фильтр
def aperture_func(nx, fx, fy, delta): 
    res = np.abs(np.sinc(delta*fx)*np.sinc(delta*fy))**2
    res[np.isnan(res)] = 0
    return res
    
# --- модуль |f|^11/3 
def abs_f(nx, fx, fy):
    res = pow(np.sqrt(fx**2+fy**2), -11./3.)
    res[np.isnan(res)] = 0
    return res

# --- фильтр Френеля
def ffilter(nx, fx, fy, z, lambda_): 
    res = pow(np.sin(np.pi*z*lambda_*(fx**2+fy**2)), 2) / pow(lambda_, 2)
    res[np.isnan(res)] = 0
    return res

# --- монохроматические гаммы со скоростями 0 м/с
def gamma_mono(z, lambda_, cjk=None, const2=None, nx=None, fx=None, fy=None, Aff113=None):
    Fresnel_filter = ffilter(nx, fx, fy, z, lambda_)   
    
    res = Aff113 * Fresnel_filter
    
    res[np.isnan(res)] = 0
    
    # (!) занимает 0.02 времени, думаю, можно научить делать fft сразу для 3d массива, чтобы в цикле его не считать
    res = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(res), s=res.shape, norm='backward'))
    
    res = res * const2
    res = res * cjk
    return res


def two(lambda_, cjk=None, D=None, sec_per_frame=None, latency=None):  
    print('Creating temporary files...')
    st = time.perf_counter()
   
    nx=cjk.shape[0]
    f_scale = 1/(2*D)
    delta = D/(cjk.shape[0]//2)
    fx, fy = f_scale * np.asarray(np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx)))
    
    f_11_3 = abs_f(nx, fx, fy)
    A_f = aperture_func(nx, fx, fy, delta) 
    Aff113 = f_11_3 * A_f
    
    Cn2 = 1e-13
    const = 9.69*pow(10, -3)*16*pow(np.pi, 2)
    const2 = const * Cn2 * pow(f_scale*nx, 2)
    
    k=50
    a1 = np.linspace(0, 50000, k)
    gammas1 = np.zeros((k, cjk.shape[0], cjk.shape[1]), dtype=np.float32)
    
    st2 = time.perf_counter()
    for i in range(k):
        tmp = gamma_mono(a1[i], lambda_, cjk=cjk, const2=const2, nx=nx, fx=fx, fy=fy, Aff113=Aff113)
#         tmp = tmp * cjk
        gammas1[i] = gaussian(tmp, sigma=1)
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - {k} turbulence layers from 0 to 50 km')
    return gammas1