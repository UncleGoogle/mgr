import sys
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from scipy import fftpack
from skimage import restoration

from helpers.readImage import readImage

# -------------------------------------Arguments & Settings-------------------

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help="Blured image")
parser.add_argument('psf', type=str, help="PSF of the corresponding image")
parser.add_argument('--output', type=str, help="where to store results")
parser.add_argument('--kowalsky', type=float, help="factor to enhance divition",
                    default=0.0001)

args = parser.parse_args()
output_path = pathlib.Path(args.output).resolve() if args.output else None
kowalsky = args.kowalsky

# -------------------------------------reading images-------------------------

u_in = readImage(args.image, color_mode=0)  # 0=grayscale
h_in = readImage(args.psf, color_mode=0)  # PSF

# =====================================comment out for testing sth new========

# -------------------------------------computing FFT--------------------------

# H_in = (np.fft.fft2(h_in))
# U_in = (np.fft.fft2(u_in))
# H_magnitude = 20*np.log(np.abs(H_in+kowalsky))
# U_magnitude = 20*np.log(np.abs(U_in+kowalsky))

# # -------------------------------------dividing FFTs-------------------------
# U_deblur = np.divide(U_in, H_in + kowalsky)
# U_deblur_show = 20*np.log(np.abs(U_deblur))
# # phase = np.arctan2(U_deblur.imag, U_deblur.real)

# -------------------------------------inverse FFT----------------------------

# u_deblur = (np.fft.ifft2(np.fft.ifftshift(U_deblur)))
# u_deblur_show = 2*np.abs(u_deblur)
# # u_deblur_show = u_deblur_show / max(np.ndarray.flatten(u_deblur_show))


# # -------------------------------------DRAWING RESULTS------------------------
# DPI=96
# plt.figure(figsize=(1200/DPI, 800/DPI), dpi=DPI)

# plt.subplot(231)
# plt.imshow(u_in, cmap='gray')
# plt.title('u')

# plt.subplot(232);
# plt.imshow(h_in, cmap='gray')
# plt.title('PSF')

# plt.subplot(234);
# plt.imshow(U_magnitude, cmap='gray')
# plt.title('U')

# plt.subplot(235);
# plt.imshow(H_magnitude, cmap='gray')
# plt.title('OTF')

# plt.subplot(236);
# plt.imshow(U_deblur_show, cmap='gray')
# plt.title('U/OTF')

# plt.subplot(233);
# plt.imshow(u_deblur_show, cmap='gray');
# plt.title('u_deblured = iFT(U/OTF)')

# if output_path:
#     plt.savefig(output_path, bbox_inches='tight', dpi=DPI*2)
# else:
#     plt.show()

# ============================================================================
# -------------------------------------scipack--------------------------------

# def convolve(star, psf):
#     star_fft = fftpack.fftshift(fftpack.fftn(star))
#     psf_fft = fftpack.fftshift(fftpack.fftn(psf))
#     return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))
# def deconvolve(star, psf):
#     star_fft = fftpack.fftshift(fftpack.fftn(star))
#     psf_fft = fftpack.fftshift(fftpack.fftn(psf))
#     return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))
# u_deblur = deconvolve(u_in, h_in)
# f, axes = plt.subplots(2,2)
# axes[0,0].imshow(u_in)
# axes[0,1].imshow(h_in)
# axes[1,0].imshow(np.real(u_deblur))
# plt.show()

# =====================================Lucy-Richardson========================



# -------------------------------------artificial-image--------------------------
# from scipy.signal import convolve2d
# from skimage import color, data, restoration
# astro = color.rgb2gray(data.astronaut())
# psf = np.ones((5, 5)) / 25
# astro = convolve2d(astro, psf, 'same')
# # Add Noise to Image
# astro_noisy = astro.copy()
# astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
# deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
# plt.gray()

# for a in (ax[0], ax[1], ax[2]):
#        a.axis('off')

# ax[0].imshow(astro)
# ax[0].set_title('Original Data')

# ax[1].imshow(astro_noisy)
# ax[1].set_title('Noisy data')

# ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
# ax[2].set_title('Restoration using\nRichardson-Lucy')
# fig.subplots_adjust(wspace=0.02, hspace=0.2,
#                     top=0.9, bottom=0.05, left=0, right=1)
# plt.show()

# -------------------------------------my-case--------------------------------

# kowalscy = [0.01, 0.001, 0.0001]
# for kow in kowalscy:
    

iterations = [1,5,15]
deconvolved_RL = {}
for i in iterations:
    deconvolved_RL[i] = restoration.richardson_lucy(u_in+kowalsky, h_in+kowalsky, iterations=i)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(9, 4))
plt.gray()
plt.suptitle('Richardson-Lucy deconvolution', fontsize=16)

ax[0].axis('off')
ax[0].imshow(u_in)
ax[0].set_title('Original Data:\ndefocus: 10mm')

ax[1].axis('off')
ax[1].imshow(h_in)
ax[1].set_title('PSF')
for i, a in enumerate(ax[2:]):
    a.axis('off')
    a.imshow(deconvolved_RL[iterations[i]])
    a.set_title(f'{iterations[i]} iterations')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

plt.show()

