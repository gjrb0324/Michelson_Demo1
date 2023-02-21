from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cv2
import numpy as np
from mpmath import *

lam = 523*pow(10,-9)
pixel = 2.4*pow(10,-6)
pixel_freq = 1/pixel

img = cv2. imread('./data/1.bmp',cv2.IMREAD_GRAYSCALE)
img = img[500:2400,500:2400]
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fftd = 20*np.log(np.abs(fshift))
fftd[949:952,949:952]=0
ishift = np.fft.ifftshift(fftd)
argmax_fftd = divmod(np.argmax(ishift), np.shape(ishift)[1])
sin_xz,sin_yz =  np.multiply(pixel_freq*lam/(2*pi),argmax_fftd)
print(sin_xz,sin_yz)
theta_xz = csc(sin_xz)
theta_yz = csc(sin_yz)
print(theta_xz,theta_yz)
'''
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Input'), plt.xticks([])
plt.yticks([])
plt.subplot(132),plt.imshow(fftd, cmap='gray')
plt.title('Fourier transformed- Shifted')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(ishift, cmap='gray')
plt.title('Fourier transformed')
plt.xticks([]),plt.yticks([])
plt.show()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.arange(0,5000,1)
Y = np.arange(0,5000,1)
X, Y = np.meshgrid(X, Y)
Z = np.pad(fftd, ((1000,1000),(500,500)), 'constant')
surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(0,400)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

fig.colorbar(surf,shrink=0.5, aspect=5)

plt.show()
'''
