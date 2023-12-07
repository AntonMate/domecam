import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import shift
from scipy.optimize import curve_fit

from skimage.filters import gaussian

def circle(radius, size, circle_centre=(0, 0), origin="middle"):
    """
    Create a 2-D array: elements equal 1 within a circle and 0 outside.

    The default centre of the coordinate system is in the middle of the array:
    circle_centre=(0,0), origin="middle"
    This means:
    if size is odd  : the centre is in the middle of the central pixel
    if size is even : centre is in the corner where the central 4 pixels meet

    origin = "corner" is used e.g. by psfAnalysis:radialAvg()

    Examples: ::

        circle(1,5) circle(0,5) circle(2,5) circle(0,4) circle(0.8,4) circle(2,4)
          00000       00000       00100       0000        0000          0110
          00100       00000       01110       0000        0110          1111
          01110       00100       11111       0000        0110          1111
          00100       00000       01110       0000        0000          0110
          00000       00000       00100

        circle(1,5,(0.5,0.5))   circle(1,4,(0.5,0.5))
           .-->+
           |  00000               0000
           |  00000               0010
          +V  00110               0111
              00110               0010
              00000

    Parameters:
        radius (float)       : radius of the circle
        size (int)           : size of the 2-D array in which the circle lies
        circle_centre (tuple): coords of the centre of the circle
        origin (str)  : where is the origin of the coordinate system
                               in which circle_centre is given;
                               allowed values: {"middle", "corner"}

    Returns:
        ndarray (float64) : the circle array
    """
    # (2) Generate the output array:
    C = np.zeros((size, size), dtype=np.float32)

    # (3.a) Generate the 1-D coordinates of the pixel's centres:
    # coords = numpy.linspace(-size/2.,size/2.,size) # Wrong!!:
    # size = 5: coords = array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])
    # size = 6: coords = array([-3. , -1.8, -0.6,  0.6,  1.8,  3. ])
    # (2015 Mar 30; delete this comment after Dec 2015 at the latest.)

    # Before 2015 Apr 7 (delete 2015 Dec at the latest):
    # coords = numpy.arange(-size/2.+0.5, size/2.-0.4, 1.0)
    # size = 5: coords = array([-2., -1.,  0.,  1.,  2.])
    # size = 6: coords = array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])

    coords = np.arange(0.5, size, 1.0, dtype=np.float32)
    # size = 5: coords = [ 0.5  1.5  2.5  3.5  4.5]
    # size = 6: coords = [ 0.5  1.5  2.5  3.5  4.5  5.5]

    # (3.b) Just an internal sanity check:
    if len(coords) != size:
        raise exceptions.Bug("len(coords) = {0}, ".format(len(coords)) +
                             "size = {0}. They must be equal.".format(size) +
                             "\n           Debug the line \"coords = ...\".")

    # (3.c) Generate the 2-D coordinates of the pixel's centres:
    x, y = np.meshgrid(coords, coords)

    # (3.d) Move the circle origin to the middle of the grid, if required:
    if origin == "middle":
        x -= size / 2.
        y -= size / 2.

    # (3.e) Move the circle centre to the alternative position, if provided:
    x -= circle_centre[0]
    y -= circle_centre[1]

    # (4) Calculate the output:
    # if distance(pixel's centre, circle_centre) <= radius:
    #     output = 1
    # else:
    #     output = 0
    mask = x * x + y * y <= radius * radius
    C[mask] = 1

    # (5) Return:
    return C

def processApprox(cc=None, gammas=None, lambda_=None, D=None, latency=None, sec_per_frame=None, cjk=None, initial_params=None, all_Vx=None, all_Vy=None, all_Cn2_bounds=None, conjugated_distance=None, num_of_layers=None, heights_of_layers=None, dome_index=None, use_gradient=False, do_fitting=True):
    print(' - initial guess for the parameters:')
    df_ip = pd.DataFrame(initial_params, columns = ['Vx, m/s','Vy, m/s','Cn2', 'z, m']) 
    df_ip = df_ip.sort_values(by=['z, m'])
    df_ip = df_ip.reset_index()
    df_ip['Cn2'] = df_ip['Cn2']*1e-13
    df_ip['z, m'] = df_ip['z, m']*1000
    df_ip = df_ip.round({'Vx, m/s': 2})
    df_ip = df_ip.round({'Vy, m/s': 2})
    df_ip = df_ip.round({'z, m': 2})
    df_ip.drop(columns=['index'], inplace=True)
    print(df_ip.to_string(index=False))
       
    def fitconvert(fit1d,shape):
        n_elem=shape[1]*shape[2]
        fit = np.ndarray(shape)
        for idx in range(len(latency)):
            tmp = fit1d[idx*n_elem:(idx+1)*n_elem]
            fit[idx] = tmp.reshape((shape[1],shape[2]))
        return fit

    def find_nearest(array, value):
        # array = np.asarray(array)
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

    def gamma_se(X, Y, t_delta, Vx, Vy, Cn2, z): 
#        Cn2=Cn2*1e-13 # БС: это не обязательно делать, см. комментарий в конце этой функции
        z=z*1000

        uv, lv = find_nearest(heights_of_layers, z)
        res = gammas[lv] + (z - heights_of_layers[lv])*((gammas[uv] - gammas[lv])/(heights_of_layers[uv] - heights_of_layers[lv]))
        
        Xpix = Vx*t_delta
        Ypix = Vy*t_delta
        res = shift(res, (-Ypix, Xpix), order=1)  
# БС: здесь был баг - не было умножения на Cn2, из-за чего Cn2 вообще не подбиралось
        return res*Cn2 

    def one_speckle_fit(initial_params=None, data=None, latency=None, lambda_=None, all_Vx=None, all_Vy=None, all_Cn2_bounds=None, conjugated_distance=None, dome_index=None, use_gradient=False, do_fitting=True): 
        class _g:

            def __init__(self):
                pass

            def fitfun(self, M, *args):
                xi, yi = M

                n_elem = int(xi.shape[0]/len(latency))
                array_size = int(np.sqrt(n_elem))
                x = xi[:n_elem]
                y = yi[:n_elem]

# БС: Собираем модельную кросс-корреляцию для разных задержек
                total_cc = np.zeros(xi.shape)
                for idx in range(len(latency)):
                    delta = D/(array_size//2)
                    t_delta = latency[idx] * sec_per_frame/delta

                    if self.use_gradient: 
                        # БС: Модификация функции _g, моделирующая градиенты между турбулентными слоями
                        # путем кусочной линейной интерполяции 
                        arr = np.zeros(x.shape, dtype=np.float32)
                        Nsublayer = 15
                        arr += gamma_se(x, y, t_delta, args[0], args[1], args[2], args[3]).ravel() # БС: между первым и вторым слоем градиент не делаем
                        for i in range(1,len(args)//4-1):
                            Vx_range  = np.linspace(args[i*4],  args[i*4+4],Nsublayer)
                            Vy_range  = np.linspace(args[i*4+1],args[i*4+5],Nsublayer)
                            Cn2_range = np.linspace(args[i*4+2],args[i*4+6],Nsublayer)
                            z_range   = np.linspace(args[i*4+3],args[i*4+7],Nsublayer)
                            for j in range(Nsublayer):
#                                Vsigma=z_range[j]/7
                                term = gamma_se(x, y, t_delta, Vx_range[j], Vy_range[j], Cn2_range[j]/Nsublayer, z_range[j])
#                                arr += gaussian(term, sigma=Vsigma*t_delta).ravel()
                                arr += term.ravel()
                    else:
                        arr = np.zeros(x.shape, dtype=np.float32)
                        for i in range(len(args)//4):
                            arr += gamma_se(x, y, t_delta, args[i*4], args[i*4+1], args[i*4+2], args[i*4+3]).ravel()
                    total_cc[idx*n_elem:(idx+1)*n_elem] = arr
                if hasattr(self,'mask'):
                    total_cc *= self.mask.ravel()
                return total_cc

        p0 = [p for prms in initial_params for p in prms]

        x = np.linspace(-data.shape[2]//2, data.shape[2]//2-1, data.shape[2], dtype=np.float32)
        y = np.linspace(-data.shape[1]//2, data.shape[1]//2-1, data.shape[1], dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        # mask = np.zeros(data.shape)
        # mask[0] = (X>-10)*(X<10)*(Y<10)*(Y>-10)
        # data *= mask

        xdata = np.vstack((X.ravel(), Y.ravel())) 
        ydata = data.ravel()

        xdata = np.tile(xdata,len(latency))

        # более точные баунсы для начальных параметров
        if do_fitting:
            lb2 = np.zeros((len(p0)//4, 4), dtype=np.float32)
            ub2 = np.zeros((len(p0)//4, 4), dtype=np.float32)
            for i in range(len(all_Vx)):
                if  i == dome_index:
                    lb2[i] = [all_Vx[i]-0.5, all_Vy[i]-0.5, all_Cn2_bounds[i][0]-0.005, conjugated_distance-(conjugated_distance*0.02)]
                    ub2[i] = [all_Vx[i]+0.5, all_Vy[i]+0.5, all_Cn2_bounds[i][1]+0.005, conjugated_distance+(conjugated_distance*0.02)]
                else:
                    lb2[i] = [all_Vx[i]-0.5, all_Vy[i]-0.5, all_Cn2_bounds[i][0]-0.005, conjugated_distance]
                    ub2[i] = [all_Vx[i]+0.5, all_Vy[i]+0.5, all_Cn2_bounds[i][1]+0.005, 50]
            lb2=np.ravel(lb2)
            ub2=np.ravel(ub2)

#         lb = [-np.inf, -np.inf, 0, conjugated_distance]
#         lb = np.tile(lb, len(p0)//4) 
#         ub = [np.inf, np.inf, np.inf, 50]
#         ub = np.tile(ub, len(p0)//4)

# БС: Класс позволяет передавать аппроксимирующей функции дополнительные параметры, 
# в данном случае используем ли мы градиент между слоями или нет.
# Такая реализация лучше, поскольку мы концентрируем сложение слоев в одной функции _g.fitfun
        _g = _g()
        _g.use_gradient = use_gradient
        _g.latency = latency
#         _g.mask = mask

# БС: считаем невязку для начального приближения
        fit_p0 = fitconvert(_g.fitfun(xdata,*p0),data.shape)

        residual_p0 = np.sum(np.power(data-fit_p0,2))
        print(f'residual for initial guess:{residual_p0}')

        if do_fitting:
            popt, pcov = curve_fit(_g.fitfun, xdata, ydata, p0, bounds=[lb2, ub2])   
        else:
            popt = np.array(p0)

        fit = fitconvert(_g.fitfun(xdata,*popt),data.shape)

#        for i in range(len(popt)//4):
#            fit += gamma_se(X, Y, *popt[i*4:i*4+4])

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
        print(' - found params:')
        print(df.to_string(index=False))
        print(' - total Cn2:', sum_cn2)
        print(f' - seeing, {lambda_/1e-9:.0f} nm: {seeing:.2f}')

# БС: считаем невязку
        print(fit.shape)
        mask = np.zeros(fit.shape)
        residual = np.sum(np.power(data-fit,2))
        print(f'total residual:{residual}')

        return fit
    
    # изображение кросс-корреляции делится на cjk, чтобы в аппроксимации не учитывать его домножение

    # fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
    # cc=cc/cjk
    # cc[np.isinf(cc)] = 0
    # fig.colorbar(ax.imshow(cc), ax=ax)
    # fig.colorbar(ax2.imshow(circle(cc.shape[0]//2-3,cc.shape[0])), ax=ax2)
    # cc = cc * circle(cc.shape[0]//2-3,cc.shape[0])
    # # Для составления изображения весов для кервфита можно будет за пределами круга поменять 0 на 1е8 
    # fig.colorbar(ax3.imshow(cc), ax=ax3)
    # ax.set_title('cc/cjk')
    # ax2.set_title('circle')
    # ax3.set_title('cc/cjk * circle')
    
    fit = one_speckle_fit(initial_params=initial_params, data=cc, latency=latency, lambda_=lambda_, all_Vx=all_Vx, all_Vy=all_Vy, all_Cn2_bounds=all_Cn2_bounds, conjugated_distance=conjugated_distance, dome_index=dome_index, use_gradient=use_gradient, do_fitting=do_fitting)
    


    return fit
#         print()
#         return fit, popt
