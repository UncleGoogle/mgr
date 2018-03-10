import sys
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

root_path = pathlib.Path(__file__).absolute().parents[1]
sys.path.insert(1, str(root_path))
from helpers import readImages, Boardspace

# -------------------------------------settings-------------------------------
show = False
write = True

# -------------------------------------reading images-------------------------

input_path = root_path / 'sample_data'
output_path = root_path / 'sample_results'

filenames = [str(name) for name in input_path.iterdir()]
u_in, h_in = readImages(filenames, color_mode=0)  # 0=grayscale

# -------------------------------------computing FFT--------------------------

kowalsky = 0.001
H_in = np.fft.fftshift(np.fft.fft2(h_in))
U_in = np.fft.fftshift(np.fft.fft2(u_in))
H_magnitude = 20*np.log(np.abs(H_in+kowalsky))
U_magnitude = 20*np.log(np.abs(U_in+kowalsky))

plt.subplot(221)
plt.imshow(u_in, cmap='gray')
plt.title('u')

plt.subplot(222);
plt.imshow(h_in, cmap='gray')
plt.title('PSF')

plt.subplot(223);
plt.imshow(U_magnitude, cmap='gray')
plt.title('U magnitude')

plt.subplot(224);
plt.imshow(H_magnitude, cmap='gray')
plt.title('OTF magnitude')

if show:
    plt.show()
elif write:
    plt.savefig(str(output_path / 'input_and_FFTs.png'), bbox_inches='tight', dpi=130)

# -------------------------------------dividing FFTs-------------------------

U_deblur = np.divide(U_in, H_in + kowalsky)
U_deblur_magnitude = 20*np.log(np.abs(U_deblur))

plt.subplot(121);
plt.imshow(U_deblur_magnitude, cmap='gray')
plt.title('20*log(abs(U/H + kowalsky))')

# -------------------------------------inverse FFT----------------------------

u_deblur = np.fft.ifft2(np.fft.ifftshift(U_deblur))
u_deblur_magnitude = 20*np.log(np.abs(u_deblur))

plt.subplot(122);
plt.imshow(u_deblur_magnitude, cmap='gray');
plt.title('20*log(abs(iFFT(U/H+kowalsky))')

if show:
    plt.show()
elif write:
    plt.savefig(str(output_path / 'deblured.png'), bbox_inches='tight', dpi=130)

