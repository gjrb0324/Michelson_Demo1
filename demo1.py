from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cv2
import numpy as np
from math import *

'''Constants Defined'''
lam = 532*pow(10,-9)*1000 #1000 for mm scale
pixel = 1.85*pow(10,-6)*1000*3000 #mm scale
pixel_freq = 1/pixel #frequency domain:1/mm*1/mm scale,
#(x,y) on frequency domain: x*pixel_freq times oscilate within 1mm.

'''Function Region'''
def calc_theta_diff(number, file_a, file_f):

    '''Proceed 2DFFT, calculate angle, and save their results'''

    im_name = './data/'+str(number)+'.bmp'
    img = cv2. imread(im_name,cv2.IMREAD_GRAYSCALE)
    img = img[0:3000,0:3000]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fftd = 20*np.log(np.abs(fshift))
    fftd[1497:1504,1497:1504]=0
    fftd = fftd[1425:1576, 1425:1576]
    ishift = np.fft.ifftshift(fftd)
    argmax_fftd = divmod(np.argmax(ishift), np.shape(ishift)[1])

    sin_xz,sin_yz =  np.multiply(lam*pixel_freq,argmax_fftd)
    theta_xz = degrees(np.arcsin(sin_xz))
    theta_yz = degrees(np.arcsin(sin_yz))
    inter_freq = sqrt(sin_xz*sin_xz + sin_yz*sin_yz)/lam
    d_lambda = 1/inter_freq

    file_a.write(str(number)+' - gamma1: '+str(theta_xz)+' gamma2: '+str(theta_yz))
    file_f.write(str(number)+' - interfered frequency: '+str(inter_freq) \
               + 'mm^-1, interfered_wavelength: '+str(d_lambda))
    file_a.write('\n')
    file_f.write('\n')

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Input'), plt.xticks([])
    plt.yticks([])
    plt.subplot(122),plt.imshow(fftd, cmap='gray')
    plt.title('Fourier transformed- Shifted')
    plt.xticks([]),plt.yticks([])

'''Actual Part'''
file_angle = open("angle.txt","a")
file_freq = open("freq_and_wavelength.txt","a")
for num in range(1,11):
    calc_theta_diff(num, file_angle,file_freq)

file_angle.close()
file_freq.close()
