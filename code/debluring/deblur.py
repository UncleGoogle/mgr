import sys
import pathlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

sys.path.insert(1, str(pathlib.Path(__file__).parent / '..'))
from helpers import readImages, Boardspace


# -------------------------------------reading images-------------------------

root_path = pathlib.Path(__file__).parents[1].absolute()
input_path = root_path / 'sample_data'
output_path = root_path / 'sample_results'

filenames = [str(name) for name in input_path.iterdir()]
h_in, u_in = readImages(filenames)

# -------------------------------------computing FFT--------------------------

kowalsky = 0.001
H_in, U_in = list(map(lambda x: np.fft.fftshift(x), (h_in, u_in)))
U_magnitude = 20*np.log(np.abs(U_in+kowalsky))

plt.subplot(121);
plt.imshow(u_in, cmap='gray');
plt.subplot(122);
plt.imshow(U_magnitude, cmap='gray')
plt.savefig(str(output_path / 'u_in and U_in.png'), bbox_inches='tight', dpi=130)

# -------------------------------------deconvolution by dividing FFTs---------

U_deblur = np.divide(U_in, H_in + kowalsky)
u_deblur = np.fft.ifft2(np.fft.ifftshift(U_deblur))
u_deblur_magnitude = 20*np.log(np.abs(u_deblur))

plt.subplot(111);
plt.imshow(u_deblur_magnitude, cmap='gray');
plt.savefig(str(output_path / 'deblured_broken.png'), bbox_inches='tight', dpi=130)


