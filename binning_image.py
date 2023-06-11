import numpy as np
from astropy.io import fits


'''
функция бинирования изображения
'''

def binning(image, factor=None):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    # https://scipython.com/blog/binning-a-2d-array-in-numpy/
    
    image = image[:-(image.shape[0] - factor*(image.shape[0]//factor)), :-(image.shape[1] - factor*(image.shape[1]//factor))]
    res = image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor)
    res = res.sum(1).sum(2)
    return res

def binbin(file, bin_factor=None, data_dir=None):
    print('Binning...')
    
    with fits.open("".join([data_dir, '/', file])) as f:
        data = np.float32(f[0].data)
        
    res = np.array([binning(image, factor=bin_factor) for image in data])
    fits.writeto(f'{data_dir}/binned{file}', res, overwrite=True)
    print(f' - Done! New shape: {res.shape}')