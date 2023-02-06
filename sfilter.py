import numpy as np

def sfilter(coeff, lambda_max=None, f_lambda=None, D=None, Dpix=None, z=None):
    k = 1000*coeff
    lambda_max_new = int(lambda_max*coeff)
    lambdas = np.linspace(0, lambda_max_new, k) * pow(10, -9) # [м]

    tail = np.zeros((len(lambdas) - len(f_lambda)))
    f_lambda_new = np.append(f_lambda, tail)
    res_fft = pow((np.imag(np.fft.fft(f_lambda_new/lambdas))), 2)

    delta_lambdas = (lambda_max_new / len(lambdas)) * pow(10, -9) # период дискретизации, шаг по частоте [м]
    omega_lambdas_scale = 1 / (delta_lambdas) # максимальное значение по частоте, [м^-1]

    nx = 2*Dpix+1 # размер окна, [n_pix]
    delta = D/Dpix # шаг субапертуры, период дискретизации (то, насколько одно значение отстает от следующего) [м]
    f_scale = 1/(delta*nx) # шаг по частоте, [м^-1]

    xx, yy = np.meshgrid(np.linspace(-nx//2, nx//2-1, nx), np.linspace(-nx//2, nx//2-1, nx))
    xx_scale = f_scale * xx 
    yy_scale = f_scale * yy
    f_abs = np.sqrt(pow(xx_scale, 2) + pow(yy_scale, 2))

    omega = 0.5 * z * pow(f_abs, 2) # аргумент, шаг по частоте, [м^-1]
    omega = np.ravel(omega)
    omega_new = np.interp(omega, np.linspace(0, omega_lambdas_scale, k), res_fft)
    omega_new = np.resize(omega_new, (nx, nx))
    
    return omega_new