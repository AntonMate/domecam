#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def cross_corr(img1, img2):
    corr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(img1)*np.fft.fft2(img2).conjugate())))
    corr /= np.max(corr)
    return corr

