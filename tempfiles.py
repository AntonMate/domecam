import time
import numpy as np
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import xlrd
import csv

np.seterr(divide='ignore')

# --- апертурный фильтр
def processApertureFilter(nx, fx, fy, delta): 
    res = np.abs(np.sinc(delta*fx)*np.sinc(delta*fy))**2
    res[np.isnan(res)] = 0
    return res
    
# --- модуль |f|^11/3 
def processF_113(nx, fx, fy):
    res = pow(np.sqrt(fx**2+fy**2), -11./3.)
    res[np.isnan(res)] = 0
    return res

# --- фильтр Френеля
def processFresnelFilter(nx, fx, fy, z, lambda_): 
    res = pow(np.sin(np.pi*z*lambda_*(fx**2+fy**2)), 2) / pow(lambda_, 2)
    res[np.isnan(res)] = 0
    return res

# --- монохроматические гаммы со скоростями 0 м/с
def processGammaMono(z, lambda_, cjk=None, const2=None, nx=None, fx=None, fy=None, Aff113=None):
    Fresnel_filter = processFresnelFilter(nx, fx, fy, z, lambda_)   
    
    with np.errstate(invalid='ignore'):
        res = Aff113 * Fresnel_filter
    
    res[np.isnan(res)] = 0
    
    # (!) занимает 0.02 времени, думаю, можно научить делать fft сразу для 3d массива, чтобы в цикле его не считать
    res = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(res), s=res.shape, norm='backward'))
    
    res = res * const2
    res = res * cjk
    return res

# --- получение кривой пропускания фильтра
def filter_values(file):  
    def xls_to_csv(file):
        x =  xlrd.open_workbook(file)
        x1 = x.sheet_by_name('Измерение')
        csvfile = open('filter.csv', 'w')
        writecsv = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        for rownum in range(1, x1.nrows): # пропускаю первую сртоку в файле, тк. там текст
            writecsv.writerow(x1.row_values(rownum))

        csvfile.close()
        
    xls_to_csv(file)
    
    x_filter, y_filter = [], []
    with open('filter.csv') as f1:
        for line in f1:
            line = line.replace('"', '').strip().split(sep=',')
            if len(line) == 2:
                line = np.array(line, dtype=np.float32)
                x_filter.append(line[0])
                y_filter.append(line[1]/100)
    return x_filter, y_filter

# --- кривая квантовой эффективности детектора
def ccd_values(file):
    with open(file) as f1, open(file) as f2:
        x_ccd = [int(np.array(line.strip().split(' '), float)[0]*1000) for line in f1]
        x_ccd = x_ccd[:len(x_ccd)-1] # последнее значение по длине волны почему-то 0, тут я его удаляю
    
        y_ccd = []
        for line in f2:
            if len(np.array(line.strip().split(' '), dtype=np.float32)) == 2:
                y_ccd.append(np.array(line.strip().split(' '), dtype=np.float32)[1])
                
    return x_ccd, y_ccd

# --- излучение звезды
def star_values(file):
    with open(file) as f1, open(file) as f2:
        x_star, y_star = [], []
        for line in f1:
            line = line.strip().split(' ')
            del line[1]        
            x_star.append(int(np.array(line[0], float)))
            y_star.append(np.array(line[1], float))   
    return x_star, y_star

# --- функция спектрального отклика
def processF_lamda(file_star=None, file_filter=None, file_ccd=None):
    x_filter, y_filter = filter_values(file_filter)
    x_ccd, y_ccd = ccd_values(file_ccd)
    x_star, y_star = star_values(file_star)

    x_max = int(np.min([x_ccd[-1], x_filter[-1], x_star[-1]]))
    x_min = np.max([x_ccd[0], x_filter[0], x_star[0]])

    lambdas = np.linspace(0, x_max, x_max+1, dtype=int)

    interp_ccd = np.interp(lambdas, x_ccd, y_ccd)
    interp_filter = np.interp(lambdas, x_filter, y_filter)
    interp_star = np.interp(lambdas, x_star, y_star)
    f_lambda = interp_ccd * interp_filter * interp_star
    f_lambda = f_lambda / np.sum(f_lambda) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(y_ccd, label='ccd')
    ax1.plot(y_filter, label='filter')
    ax1.plot(y_star, label='star')
    ax1.legend()
    ax1.set_title('Кривые пропускания детектора, фильтра и звезды')

    ax2.plot(f_lambda, label='F(λ)')
    ax2.legend()
    ax2.set_title('Функция спектрального отклика')
    ax2.set_xlabel('Длина волны, [нм]')
    return f_lambda

# --- спектральный фильтр
def processSpectralFilter(f_lambda=None, z=None, D=None, D_pix=None, f_abs=None):
#     z - дистанция распространения, [м]
#     D - диаметр телескопа, [м]
#     D_pix - диаметр телескопа в пикселях, [n_pix]
    coeff=100
    k = 1000*coeff
    lambda_max = len(f_lambda) - 1
    lambda_max_new = int(lambda_max*coeff)
    lambdas = np.linspace(0, lambda_max_new, k) * pow(10, -9) # [м]

    tail = np.zeros((len(lambdas) - len(f_lambda)))
    f_lambda_new = np.append(f_lambda, tail)
    with np.errstate(invalid='ignore'):
        res_fft = pow((np.imag(np.fft.fft(f_lambda_new/lambdas))), 2)

    delta_lambdas = (lambda_max_new / len(lambdas)) * pow(10, -9) # период дискретизации, шаг по частоте [м]
    omega_lambdas_scale = 1 / (delta_lambdas) # максимальное значение по частоте, [м^-1]
    

    omega = f_abs * z
    omega = np.ravel(omega)
    omega_new = np.interp(omega, np.linspace(0, omega_lambdas_scale, k), res_fft)
    omega_new = np.resize(omega_new, (2*D_pix, 2*D_pix))    
    return omega_new

def processGammaPoly(z, f_lambda=None, cjk=None, D=None, const2=None, Aff113=None, f_abs=None):
    omega_new = processSpectralFilter(f_lambda=f_lambda, z=z, D=D, D_pix=cjk.shape[0]//2, f_abs=f_abs)
 
    with np.errstate(invalid='ignore'):
        res = Aff113 * omega_new
    
    res[np.isnan(res)] = 0
    
    res = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(res), s=res.shape, norm='backward'))
    
    res = res * const2   
    res = res * cjk
    return res

def processGamma(lambda_, GammaType=None, cjk=None, D=None, sec_per_frame=None, latency=None, file_star=None, file_filter=None, file_ccd=None):  
    print('creating gammas')
    st = time.perf_counter()
    
    # интенисвность гамм
    Cn2 = 1e-13
    
    # частотный движ, фильтры и прочее, чтобы в цикле не считать каждый раз
    nx=cjk.shape[0]
    f_scale = 1/(2*D) # шаг по частоте, [м^-1]
    delta = D/(cjk.shape[0]//2) # шаг субапертуры, период дискретизации (то, насколько одно значение отстает от следующего) [м]
    fx, fy = f_scale * np.asarray(np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx)))
    f_11_3 = processF_113(nx, fx, fy)
    A_f = processApertureFilter(nx, fx, fy, delta) 
    Aff113 = f_11_3 * A_f
    
    # константы для рассчета гамм, чтобы в цикле не считать каждый раз
    const = 9.69*pow(10, -3)*16*pow(np.pi, 2)
    const2 = const * Cn2 * pow(f_scale*nx, 2)
    
    # выкладки для спектрального фильтра, чтобы в цикле не считать каждый раз
    f_abs = np.sqrt(pow(fx, 2) + pow(fy, 2))
    f_abs = 0.5 * pow(f_abs, 2)    
    
    # разбивка по высоте
    k=50
    a1 = np.linspace(0, 50000, k)
    
    # выкладки для 3м массива гамм
    gammas1 = np.zeros((k, cjk.shape[0], cjk.shape[1]), dtype=np.float32)
    
    if GammaType == 'mono':
        for i in range(k):
            tmp = processGammaMono(a1[i], lambda_, cjk=cjk, const2=const2, nx=nx, fx=fx, fy=fy, Aff113=Aff113)
        #         tmp = tmp * cjk
            gammas1[i] = gaussian(tmp, sigma=1)
    
    if GammaType == 'poly': 
        f_lambda=processF_lamda(file_star=file_star, file_filter=file_filter, file_ccd=file_ccd)
        for i in range(k):
            tmp = processGammaPoly(a1[i], f_lambda=f_lambda, cjk=cjk, D=D, const2=const2, Aff113=Aff113, f_abs=f_abs)
            gammas1[i] = gaussian(tmp, sigma=1)
    
    print(f' - time: {time.perf_counter() - st:.4f}')
    print(f' - {k} {GammaType}chromatic turbulence layers from 0 to 50 km')
    return gammas1