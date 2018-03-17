import sys
import pathlib
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic


# -------------------------------------Arguments & Settings-------------------

root_path = pathlib.Path(__file__).absolute().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help="Blured image")
parser.add_argument('psf', type=str, help="PSF of the corresponding image")
parser.add_argument('--output', type=str, help="where to store results")
parser.add_argument('--kowalsky', type=float, help="factor to enhance divition",
                    default=0.0001)

args = parser.parse_args()
output_path = pathlib.Path(args.output).resolve()
kowalsky = args.kowalsky

# -------------------------------------reading images-------------------------

def readImage(filename, dtype=None, color_mode=0):
    """Read image files as numpy arrays using opencv python wrapper.
    :param filename         string path pointing to image
    :param color_mode       1 - color, 0 - greyscale(default), -1 - unchanged
    :param dtype            data type of returned numpy array (ex. np.float32)
                            If any *float* type is used, convert to [0,1] range.
                            If not given original is used.
    :return                 numpy arrays of read image
    """
    im = cv2.imread(filename, color_mode)
    if im is None:
        raise OSError('image {} load failed!'.format(filename))
    # search for normalization factor (max no. to divide by)
    max_no = np.iinfo(im.dtype).max
    if im.dtype != dtype:
        im = np.asarray(im, dtype=dtype)
    if max_no and im.dtype in [np.float16, np.float32, np.float64]:
        im = im / max_no
    return im

u_in = readImage(args.image, color_mode=0)  # 0=grayscale
h_in = readImage(args.psf, color_mode=0)

# -------------------------------------computing FFT--------------------------

H_in = np.fft.fftshift(np.fft.fft2(h_in))
U_in = np.fft.fftshift(np.fft.fft2(u_in))
H_magnitude = 20*np.log(np.abs(H_in+kowalsky))
U_magnitude = 20*np.log(np.abs(U_in+kowalsky))

# -------------------------------------dividing FFTs-------------------------
U_deblur = np.divide(U_in, H_in + kowalsky)
U_deblur_show = 20*np.log(np.abs(U_deblur))
# phase = np.arctan2(U_deblur.imag, U_deblur.real)

# -------------------------------------inverse FFT----------------------------

u_deblur = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_deblur)))
u_deblur_show = 2*np.abs(u_deblur)
# u_deblur_show = u_deblur_show / max(np.ndarray.flatten(u_deblur_show))

# -------------------------------------DRAWING RESULTS------------------------
DPI=96
plt.figure(figsize=(1200/DPI, 800/DPI), dpi=DPI)

plt.subplot(231)
plt.imshow(u_in, cmap='gray')
plt.title('u')

plt.subplot(232);
plt.imshow(h_in, cmap='gray')
plt.title('PSF')

plt.subplot(234);
plt.imshow(U_magnitude, cmap='gray')
plt.title('U')

plt.subplot(235);
plt.imshow(H_magnitude, cmap='gray')
plt.title('OTF')

plt.subplot(236);
plt.imshow(U_deblur_show, cmap='gray')
plt.title('U/OTF')

plt.subplot(233);
plt.imshow(u_deblur_show, cmap='gray');
plt.title('u_deblured = iFT(U/OTF)')

if output_path:
    plt.savefig(output_path, bbox_inches='tight', dpi=DPI*2)
else:
    plt.show()
