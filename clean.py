import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import shift
from scipy.optimize import curve_fit
from skimage.filters import threshold_multiotsu, threshold_otsu
from astropy.io import fits

cjk, t, a1, gammas, delta = 0, 0, 0, 0, 0
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

def gamma_poly_se(X, Y, Vx, Vy, Cn2, z): 
    global cjk, t, a1, gammas, delta
    
    Cn2=Cn2*1e-14
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

def CLEAN(dirty, gain=None, thresh=None, niter=None, window=None, checkpointf=None, step=None, t=None, delta=None, X=None, Y=None):
    st = time.perf_counter()
    comps = np.zeros(dirty.shape)
    res = np.array(dirty) 
    cl = np.zeros(dirty.shape)
    
    res2 = np.array(dirty)
    cl2 = np.zeros(dirty.shape)

    tmp = 0
    tmp_Vx = []
    tmp_Vy = []
    all_params = []
    all_errors = []
    reason = 'empty'
    for i in range(niter):
        my, mx = np.unravel_index(np.argmax(res), res.shape)
        mval=res[my, mx]*gain
        comps[my, mx]+=mval
        
        Vy = (dirty.shape[0]//2-my)*delta/t
        Vx = -(dirty.shape[1]//2-mx)*delta/t
        
        tmp_Vx.append(Vx)
        tmp_Vy.append(Vy)
                
        if i > 5\
        and tmp_Vx[i-4] == tmp_Vx[i] and tmp_Vx[i-3] == tmp_Vx[i] and tmp_Vx[i-2] == tmp_Vx[i] and tmp_Vx[i-1] == tmp_Vx[i]\
        and tmp_Vy[i-4] == tmp_Vy[i] and tmp_Vy[i-3] == tmp_Vy[i] and tmp_Vy[i-2] == tmp_Vy[i] and tmp_Vy[i-1] == tmp_Vy[i]:
            all_params = np.array(all_params)
            all_params = all_params[:-4, :]
            all_errors = np.array(all_errors)
            all_errors = all_errors[:-4, :]
            reason = 'Fitting stucked!'
            print(f'{reason}')
            break
        
        if int(Vx) == 0 and int(Vy) == 0:
#             print('\nDome turbulence')
            p0_Cn2 = (res[my, mx]/np.max(gamma_poly_se(X, Y, Vx, Vy, 10, 2))) * 10
            myParams = [0, 0, p0_Cn2, 2]
            psf, params, errors = multi_speckle_fit(myParams, ydata=res, window=window//2, t=t, delta=delta, X=X, Y=Y)
#             print('-initial value:', res[my, mx])
            res -= psf*0.99
        else:
            p0_Cn2 = (res[my, mx]/np.max(gamma_poly_se(X, Y, Vx, Vy, 10, 15))) * 10
            myParams = [Vx, Vy, p0_Cn2, (15+5*np.sin(np.random.uniform(-np.pi, np.pi)))]  
            psf, params, errors = multi_speckle_fit(myParams, ydata=res, window=window, t=t, delta=delta, X=X, Y=Y)
#             print('-initial value:', res[my, mx])
            res -= psf*gain
        
        cl += psf*gain
        all_params.append(params)
        all_errors.append(errors)
#         print('-residual value:', res[my, mx])
        
        if step is not None and checkpointf == 'yes':
            if (i+1)%step == 0:
                print('\nCheckpoint fitting...')
                fit, params2 = one_speckle_fit(all_params[i-(step-1):i+1], res2)
                res2 -= fit
                cl2 += fit
            
#             plt.figure()
#             plt.imshow(res2)
#             plt.show()
            
#             plt.figure()
#             plt.imshow(cl2)
#             plt.show()
        
        tmp = i
        if np.max(res) < thresh:
            reason = 'Done, thresh reached!'
            print(f'{reason}')
            break
    
    if reason == 'empty':
        reason = 'Max number of iteration reached!'
        print(f'{reason}')
        
    print('Total iterations:', tmp+1)
    print('time:', time.perf_counter()-st)
    print()
    conf = [reason, tmp+1, time.perf_counter()-st]
    
    if step is None and checkpointf == 'yes':
        print('Final fitting...')
        fit, params2 = one_speckle_fit(all_params, dirty)

    return comps, res, cl, np.array(all_params), np.array(all_errors), conf


def multi_speckle_fit(params, ydata=None, window=None, t=None, delta=None, X=None, Y=None):
#     st=time.perf_counter()  
    def speckle_fit(params, ydata, window=None):
        k=window
        def _g(one_dim_x, *args): 
            arr = np.zeros(one_dim_x[0].shape)
            Vx = args[0]
            Vy = args[1]
            Cn2 = args[2]
            z = args[3]
            arr += gamma_poly_se(one_dim_x[0], one_dim_x[1], Vx, Vy, Cn2, z)[Ypix1-k:Ypix1+k, Xpix1-k:Xpix1+k].ravel()
#             arr += gamma_poly_se(one_dim_x[0], one_dim_x[1], Vx, Vy, Cn2, z).ravel()
            return arr
        
        xcoord = params[0]
        ycoord = params[1]
        cr = ydata.shape[0]//2, ydata.shape[1]//2
        
        Xpix1 = int(xcoord*t/delta) + cr[1]
        Ypix1 = -int(ycoord*t/delta) + cr[0]
        
        fit = np.zeros(ydata.shape)
        ydata = ydata[Ypix1-k:Ypix1+k, Xpix1-k:Xpix1+k]
        
#         plt.figure()
#         plt.imshow(ydata, cmap='gray')
#         plt.colorbar()
#         plt.show()
        
        x = np.linspace(-ydata.shape[1]//2, ydata.shape[1]//2-1, ydata.shape[1])
        y = np.linspace(-ydata.shape[0]//2, ydata.shape[0]//2-1, ydata.shape[0])
        X, Y = np.meshgrid(x, y)
        
        xdata = np.vstack((X.ravel(), Y.ravel()))
        
        ydata = ydata.ravel()

        bounds = [[-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]]
        popt, pcov = curve_fit(_g, xdata, ydata, p0=params, bounds=bounds)
        
        fit += gamma_poly_se(X, Y, *popt)        

        return fit, popt[0], popt[1], popt[2], popt[3], np.sqrt(np.diag(pcov))
        
    res = np.zeros(ydata.shape)
#     print(f'Speckle fitting: {params[0]} {params[1]} {1e-14*params[2]} {1000*params[3]}')
    fit, Vx, Vy, Cn2, z, errors = speckle_fit(params, ydata, window=window)
    
    xcoord = int(Vx*t/delta)
    ycoord = int(Vy*t/delta)
    res +=fit        
#     print(f'Fitted parametrs: {Vx:.2f} {Vy:.2f} {1e-14*Cn2} {1000*z}')
#     print('time:', time.perf_counter()-st)    
    return res, [Vx, Vy, Cn2, z], errors

def one_speckle_fit(params=None, data=None): 
    def _g(M, *args): 
        x, y = M
        arr = np.zeros(x.shape)
        for i in range(len(args)//4):
            arr += gamma_poly_se(x, y, *args[i*4:i*4+4]).ravel()
        return arr
        
    p0 = [p for prms in params for p in prms]
        
    st=time.perf_counter()
       
    x = np.linspace(-data.shape[1]//2, data.shape[1]//2-1, data.shape[1])
    y = np.linspace(-data.shape[0]//2, data.shape[0]//2-1, data.shape[0])
    X, Y = np.meshgrid(x, y)

    fit = np.zeros(X.shape) 
    xdata = np.vstack((X.ravel(), Y.ravel())) 
    ydata = data.ravel()
        
    lb = [-np.inf, -np.inf, 0, 0]
    lb = np.tile(lb, len(p0)//4)
    ub = [np.inf, np.inf, np.inf, np.inf]
    ub = np.tile(ub, len(p0)//4)
        
    popt, pcov = curve_fit(_g, xdata, ydata, p0, bounds=[lb, ub])

    for i in range(len(popt)//4):
        fit += gamma_poly_se(X, Y, *popt[i*4:i*4+4])

#     errors = np.sqrt(np.diag(pcov))

    popt = popt.reshape(len(popt)//4, 4)
    
    df = pd.DataFrame(popt, columns = ['Vx, m/s','Vy, m/s','Cn2, 1e-14', 'z, km'])
    df = df.sort_values(by=['z, km'])
    df = df.reset_index()
#     df_err = pd.DataFrame(errors, columns = ['Vx_err','Vy_err','Cn2_err', 'z_err'])
    
    print(df)
#     print(df_err)
    sum_cn2 = np.sum(df['Cn2, 1e-14']*1e-14)
    print()
    print('total Cn2:', sum_cn2)
    lambda_ = 500 * pow(10, -9)
    r0 = pow(0.423 * pow((2*np.pi/lambda_), 2) * sum_cn2, -3/5)
    seeing = 206265 * 0.98 * lambda_/r0
    print(f'seeing, 500 nm: {seeing:.2f}')
    print(f'Time: {time.perf_counter()-st:.4f}')
    print()
    return fit, popt


def curvef(file=None, file2=None, mode=None, D=None, latency=None, sec_per_frame=None, dist0=None, dist02=None, gain=None, thresh_manual=None, thresh_manual2=None, niter=None, window=None, run_clean=None, checkpointf=None, step=None, seeing_lambda=None, data_dir=None, save_as=None):
    global cjk, t, a1, gammas, delta
    if mode == 'orig':
        data = np.loadtxt(f'{data_dir}/{file}')
        gammas = np.load(f'{data_dir}/gammas_orig.npy')
        if file2 is not None:
            data2 = np.loadtxt(f'{data_dir}/{file2}')

    if mode == 'blur':
        data = np.loadtxt(f'{data_dir}/{file}')
        gammas = np.load(f'{data_dir}/gammas_blur.npy')
        if file2 is not None:
            data2 = np.loadtxt(f'{data_dir}/{file2}')


    pupil = np.loadtxt(f'{data_dir}/pupil.gz')
    cjk = np.loadtxt(f'{data_dir}/cjk.gz')
    a1 = np.loadtxt(f'{data_dir}/z.gz')

    # ------------ шаг для функции гамма
    x = np.linspace(-data.shape[1]//2, data.shape[1]//2-1, data.shape[1])
    y = np.linspace(-data.shape[0]//2, data.shape[0]//2-1, data.shape[0])
    X, Y = np.meshgrid(x, y)

    delta = D/(pupil.shape[0]) # шаг по пикселю
    t = latency * sec_per_frame

#     if thresh_type == 'otsu':
#         thresh = threshold_otsu(data)
#         if file2 is not None:
#             thresh2 = threshold_otsu(data2)
#     if thresh_type == 'multiotsu':
#         thresh = threshold_multiotsu(data)[0]
#         if file2 is not None:
#             thresh2 = threshold_multiotsu(data2)[0]
    thresh = threshold_multiotsu(data)[0]
    if file2 is not None:
        thresh2 = threshold_multiotsu(data2)[0]
          
    if thresh_manual is None:
        thresh_manual = 0
    if thresh_manual2 is None:
        thresh_manual2 = 0
    if dist0 is None:
        dist0 = 0
    
    
    if run_clean == 'no':
        print('Threshold settings:')
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5)) 
        fig.colorbar(ax.imshow(data, cmap='gray'), ax=ax)
        ax.set_title(f'{file}')
        ax2.imshow(data>thresh+thresh_manual, cmap='gray')
        ax2.set_title('threshold')
        plt.show()
        if file2 is not None:
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 5)) 
            fig.colorbar(ax.imshow(data2, cmap='gray'), ax=ax)
            ax.set_title(f'{file2}')
            ax2.imshow(data2>thresh2+thresh_manual2, cmap='gray')
            ax2.set_title('threshold')
            plt.show()
            
    if run_clean == 'yes':
        print(f'Fitting {file}...')
        comps, residual, clean, params, errors, conf = CLEAN(data, gain=gain, thresh=thresh+thresh_manual, niter=niter, window=window, checkpointf=checkpointf, step=step, t=t, delta=delta, X=X, Y=Y)
        df2=None
        if file2 is not None:
            print(f'Fitting {file2}...')
            comps2, residual2, clean2, params2, errors2, conf = CLEAN(data2, gain=gain, thresh=thresh2+thresh_manual2, niter=niter, window=window, checkpointf=checkpointf, step=step, t=t, delta=delta, X=X, Y=Y)
            fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5)) 
            fig.colorbar(ax.imshow(data2, cmap='gray'), ax=ax) 
            fig.colorbar(ax2.imshow(clean2, cmap='gray'), ax=ax2) 
            fig.colorbar(ax3.imshow(residual2, cmap='gray'), ax=ax3) 
            
            v = (D / pupil.shape[0]) / (latency * sec_per_frame)
            x = np.round(v*np.linspace(-data.shape[0]//2+1, data.shape[0]//2, 5), 2)
            y = np.round(v*np.linspace(-data.shape[0]//2+1, data.shape[0]//2, 5), 2)
            y = np.flipud(y)

            ax.set_xticks(np.linspace(0, data.shape[1], 5))
            ax.set_yticks(np.linspace(0, data.shape[0], 5))
            ax.set_xticklabels(x)
            ax.set_yticklabels(y)
            ax.set_ylabel('Vy, m/s')
            ax.set_xlabel('Vx, m/s')
            ax.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)

            ax2.set_xticks(np.linspace(0, data.shape[1], 5))
            ax2.set_yticks(np.linspace(0, data.shape[0], 5))
            ax2.set_xticklabels(x)
            ax2.set_yticklabels(y)
            ax2.set_ylabel('Vy, m/s')
            ax2.set_xlabel('Vx, m/s')
            ax2.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)

            ax3.set_xticks(np.linspace(0, data.shape[1], 5))
            ax3.set_yticks(np.linspace(0, data.shape[0], 5))
            ax3.set_xticklabels(x)
            ax3.set_yticklabels(y)
            ax3.set_ylabel('Vy, m/s')
            ax3.set_xlabel('Vx, m/s')
            ax3.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)

            fig.savefig(f'{data_dir}/fit2_{save_as}.png', bbox_inches='tight')
            
            ax.set_title(f'max: {np.max(data2):.4f}, min: {np.min(data2):.4f}') 
            ax2.set_title(f'max: {np.max(clean2):.4f}, min: {np.min(clean2):.4f}') 
            ax3.set_title(f'max: {np.max(residual2):.4f}, min: {np.min(residual2):.4f}')
            fig.suptitle(f'{file2}')
            
            df2 = pd.DataFrame(params2, columns = ['Vx, m/s','Vy, m/s','Cn2, 1e-14', 'z, km'])
            df2 = df2.sort_values(by=['z, km'])
            df2['z, km'] = df2['z, km'] - dist02
            df_err2 = pd.DataFrame(errors2, columns = ['err_Vx, m/s','err_Vy, m/s','err_Cn2, 1e-14', 'err_z, km'])
            print('-------------------------------------------')
            print(f'File: {file2}')
            print(df2)
            print(df_err2)
            
            sum2_cn2 = np.sum(df2['Cn2, 1e-14'])*1e-14
            print()
            print('total Cn2:', sum2_cn2)
            print('lambda, m:', seeing_lambda)
            r02 = pow(0.423 * pow((2*np.pi/seeing_lambda), 2) * sum2_cn2, -3/5)
            seeing2 = 206265 * 0.98 * seeing_lambda/r02
            print(f'seeing, arcsec: {seeing2:.2f}')
            print('-------------------------------------------')

        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5)) 
        fig.colorbar(ax.imshow(data, cmap='gray'), ax=ax) 
        fig.colorbar(ax2.imshow(clean, cmap='gray'), ax=ax2) 
        fig.colorbar(ax3.imshow(residual, cmap='gray'), ax=ax3) 
        
        v = (D / pupil.shape[0]) / (latency * sec_per_frame)
        x = np.round(v*np.linspace(-data.shape[0]//2+1, data.shape[0]//2, 5), 2)
        y = np.round(v*np.linspace(-data.shape[0]//2+1, data.shape[0]//2, 5), 2)
        y = np.flipud(y)
        
        ax.set_xticks(np.linspace(0, data.shape[1], 5))
        ax.set_yticks(np.linspace(0, data.shape[0], 5))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)
        ax.set_ylabel('Vy, m/s')
        ax.set_xlabel('Vx, m/s')
        ax.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        
        ax2.set_xticks(np.linspace(0, data.shape[1], 5))
        ax2.set_yticks(np.linspace(0, data.shape[0], 5))
        ax2.set_xticklabels(x)
        ax2.set_yticklabels(y)
        ax2.set_ylabel('Vy, m/s')
        ax2.set_xlabel('Vx, m/s')
        ax2.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        
        ax3.set_xticks(np.linspace(0, data.shape[1], 5))
        ax3.set_yticks(np.linspace(0, data.shape[0], 5))
        ax3.set_xticklabels(x)
        ax3.set_yticklabels(y)
        ax3.set_ylabel('Vy, m/s')
        ax3.set_xlabel('Vx, m/s')
        ax3.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        
        fig.savefig(f'{data_dir}/fit_{save_as}.png', bbox_inches='tight')
        
        ax.set_title(f'max: {np.max(data):.4f}, min: {np.min(data):.4f}') 
        ax2.set_title(f'max: {np.max(clean):.4f}, min: {np.min(clean):.4f}') 
        ax3.set_title(f'max: {np.max(residual):.4f}, min: {np.min(residual):.4f}')
        fig.suptitle(f'{file}')
        
        df = pd.DataFrame(params, columns = ['Vx, m/s','Vy, m/s','Cn2, 1e-14', 'z, km'])
        df = df.sort_values(by=['z, km'])
        df['z, km'] = df['z, km'] - dist0
        df_err = pd.DataFrame(errors, columns = ['err_Vx, m/s','err_Vy, m/s','err_Cn2, 1e-14', 'err_z, km'])
        print(f'File: {file}')
        print(df)
        print(df_err)
        fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5)) 

#         ax.errorbar(df['z, km'], df['Cn2, 1e-14'], yerr=df_err['err_Cn2, 1e-14'], xerr=df_err['err_z, km'], fmt='o--', color='black', label=f'{dist0}km')
#         ax2.errorbar(df['z, km'], df['Vy, m/s'], yerr=df_err['err_Vy, m/s'], xerr=df_err['err_z, km'], fmt='o--', color='black', label=f'{dist0}km')
#         ax3.errorbar(df['z, km'], df['Vx, m/s'], yerr=df_err['err_Vx, m/s'], xerr=df_err['err_z, km'], fmt='o--', color='black', label=f'{dist0}km')
        if df2 is None:
            ax.scatter(df['z, km'], df['Cn2, 1e-14'], color='black', label=f'{dist0}km')
            ax2.scatter(df['z, km'], df['Vy, m/s'], color='black', label=f'{dist0}km')
            ax3.scatter(df['z, km'], df['Vx, m/s'], color='black', label=f'{dist0}km')
            ax.plot(df['z, km'], df['Cn2, 1e-14'], color='black', ls='--')
            ax2.plot(df['z, km'], df['Vy, m/s'], color='black', ls='--')
            ax3.plot(df['z, km'], df['Vx, m/s'], color='black', ls='--')
        
        if df2 is not None:  
            ax.scatter(df['z, km'], df['Cn2, 1e-14'], color='black', label=f'{dist0}km')
            ax2.scatter(df['z, km'], df['Vy, m/s'], color='black', label=f'{dist0}km')
            ax3.scatter(df['z, km'], df['Vx, m/s'], color='black', label=f'{dist0}km')
            ax.plot(df['z, km'], df['Cn2, 1e-14'], color='black', ls='--')
            ax2.plot(df['z, km'], df['Vy, m/s'], color='black', ls='--')
            ax3.plot(df['z, km'], df['Vx, m/s'], color='black', ls='--')
        
            ax.scatter(df2['z, km'], df2['Cn2, 1e-14'], color='red', label=f'{dist02}km')
            ax2.scatter(df2['z, km'], df2['Vy, m/s'], color='red', label=f'{dist02}km')
            ax3.scatter(df2['z, km'], df2['Vx, m/s'], color='red', label=f'{dist02}km')
            ax.plot(df2['z, km'], df2['Cn2, 1e-14'], ls='--', color='red')
            ax2.plot(df2['z, km'], df2['Vy, m/s'], ls='--', color='red')
            ax3.plot(df2['z, km'], df2['Vx, m/s'], ls='--', color='red')

        
        ax.legend()
        ax2.legend()
        ax3.legend()

        ax.set_ylabel('Cn2, 1e-14')
        ax.set_xlabel('z, km')
        ax2.set_ylabel('Vy, m/s')
        ax2.set_xlabel('z, km')
        ax3.set_ylabel('Vx, m/s')
        ax3.set_xlabel('z, km')

        ax.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        ax2.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        ax3.grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)
        fig.savefig(f'{data_dir}/profile.png', bbox_inches='tight')
        
        sum_cn2 = np.sum(df['Cn2, 1e-14'])*1e-14
        print()
        print('total Cn2:', sum_cn2)
        print('lambda, m:', seeing_lambda)
        r0 = pow(0.423 * pow((2*np.pi/seeing_lambda), 2) * sum_cn2, -3/5)
        seeing = 206265 * 0.98 * seeing_lambda/r0
        print(f'seeing, arcsec: {seeing:.2f}')
             
        with open(f'{data_dir}/fitlog.txt', 'w') as f:
            print(f'file: {file}', file=f)
            print('\n-------- General settings -------', file=f)
            print(f'MODE: {mode}', file=f)
            print(f'latency: {latency}', file=f)
            print('\n-------- CLEAN settings -------', file=f)
            print(f'gain: {gain}', file=f)
            print(f'thresh: {thresh+thresh_manual}', file=f)
            print(f'niter: {niter}', file=f)
            print(f'window: {window}', file=f)
            print('\n-------- CLEAN output -------', file=f)
            print(f'message: {conf[0]}', file=f)
            print(f'num. of iters: {conf[1]}', file=f)
            print(f'time: {conf[2]}', file=f)
            print('\n-------- Results -------', file=f)
            print(f'{df}', file=f)
            print('', file=f)
            print(f'{df_err}', file=f)
            print('', file=f)
            print(f'total cn2: {sum_cn2}', file=f)
            print(f'lambda, m: {seeing_lambda}', file=f)
            print(f'seeing, arcsec: {seeing:.2f}', file=f)
            if df2 is not None:
                print('\n----------------', file=f)
                print(f'{file2}', file=f)
                print(f'{df2}', file=f)
                print('', file=f)
                print(f'{df_err2}', file=f)
                print('', file=f)
                print(f'total cn2: {sum2_cn2}', file=f)
                print(f'lambda, m: {seeing_lambda}', file=f)
                print(f'seeing, arcsec: {seeing2:.2f}', file=f) 