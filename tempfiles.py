import time
import numpy as np
from skimage.filters import gaussian

# --- апертурный фильтр
def aperture_func(nx, f_scale, delta): 
    fx, fy = f_scale * np.asarray(np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx)))
    res = np.abs(np.sinc(delta*fx)*np.sinc(delta*fy))**2
    res[np.isnan(res)] = 0
    return res
    
# --- модуль |f|^11/3 
def abs_f(nx, f_scale):
    fx, fy = f_scale * np.asarray(np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx)))
    with np.errstate(divide='ignore'):
        res = pow(np.sqrt(fx**2+fy**2), -11./3.)
    
    res[np.isnan(res)] = 0
    return res

# --- фильтр Френеля
def ffilter(nx, f_scale, z, lambda_): 
    fx, fy = f_scale * np.asarray(np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx)))
    res = pow(np.sin(np.pi*z*lambda_*(fx**2+fy**2)), 2) / pow(lambda_, 2)
    res[np.isnan(res)] = 0
    return res

# --- монохроматические гаммы со скоростями 0 м/с
def gamma_mono(Cn2, z, lambda_, cjk=None, D=None, sec_per_frame=None, latency=None):
    const = 9.69*pow(10, -3)*16*pow(np.pi, 2)
    nx=cjk.shape[0]
    f_scale = 1/(2*D)
    delta = D/(cjk.shape[0]//2)

    Fresnel_filter = ffilter(nx, f_scale, z, lambda_)
    f_11_3 = abs_f(nx, f_scale)
    A_f = aperture_func(nx, f_scale, delta) 
    
    with np.errstate(invalid='ignore'):
        res = f_11_3 * Fresnel_filter
    
    res[np.isnan(res)] = 0
    res = res * A_f
    res[np.isnan(res)] = 0
    
    res = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(res), s=res.shape, norm='backward'))
    res = res * 9.69*pow(10, -3)*16*pow(np.pi, 2)

    res = res * pow(f_scale*nx, 2)
    res = Cn2 * res   
    
    res = res * cjk
    return res


def two(lambda_, cjk=None, D=None, sec_per_frame=None, latency=None):  
    print('Creating temporary files...')
    st = time.perf_counter()
    k=50
    a1 = np.linspace(0, 50000, k)
    gammas1 = np.ndarray(shape=(k, cjk.shape[0], cjk.shape[1]), dtype=np.float32)
    for i in range(k):
        tmp = gamma_mono(1e-13, a1[i], lambda_, cjk=cjk, D=D, sec_per_frame=sec_per_frame, latency=latency)
#         gammas1[i] = tmp
        gammas1[i] = gaussian(tmp, sigma=1)
#     np.save(f'{data_dir}/tempGM500.npy', gammas1)
    
    print(f' - Done! time: {time.perf_counter() - st:.4f}')
    print(f' - {k} turbulence layers from 0 to 50 km')
    return gammas1