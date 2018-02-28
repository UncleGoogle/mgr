from uuid import uuid4
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Boardspace:
    """Displays images using opencv"""
    def __init__(self, title=str(uuid4()), x=1024, y=1024, mpl=False, option=cv2.WINDOW_NORMAL):
        """
        :param mpl          Use matplotlib; else use cv2
        """
        self.title = title
        self.mpl = mpl
        if mpl:
            self.figure = plt.figure(1, figsize=(x/100, x/100)) # size in inches
            plt.title = self.title
        else:
            cv2.namedWindow(self.window_name, option)
            cv2.resizeWindow(self.window_name, x, y)

    def show(self, images, labels, waitSeconds=0, normalize=False, output=None):
        """show images one after another
        :param images       iterable of any numerable image format
        :param labels       iterable of labels corresponding to images
        :param waitSeconds  how long image will be displayed in seconds.
                            Default `0` means: wait untill press any key
        :param normalize    boolean indicating if normalize to range [0,1].
        :param output       dont show, but save figure at given output
        """
        if not images or not len(images):
            raise SyntaxError("images should be iterable")

        for i, im in enumerate(images):
            if np.iscomplexobj(im):
                im = np.abs(im)
            if normalize:
                min_val = np.min(im.ravel())
                max_val = np.max(im.ravel())
                im = (im - min_val) / (max_val - min_val)
            if self.mpl:
                subplt = plt.subplot(len(images), 1, i+1)
                subplt.set_ylabel(labels[i], fontsize=20)
                plt.imshow(im, cmap='gray')
            else:
                cv2.imshow(self.window_name, im)
                cv2.waitKey(waitSeconds*1000)
        if self.mpl:
            if output:
                plt.savefig(output, bbox_inches='tight', dpi=90)
            else:
                plt.show()


def crop(im, startx, starty, x, y):
    return im[starty:starty+y,startx:startx+x]

def crop_center(im, cropx, cropy):
    y, x = im.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return im[starty:starty+cropy,startx:startx+cropx]

def readImages(filenames, dtype=None, color_mode=0):
    """Read image files as numpy arrays using opencv python wrapper.
    :param filenames        list of filenames
    :param color_mode       1 - color, 0 - greyscale(default), -1 - unchanged
    :param dtype            data type of returned numpy array (ex. np.float32)
                            If any *float* type is used, convert to [0,1] range.
                            If not given original is used.
    :return                 list of numpy arrays
    """

    images = []
    for name in filenames:
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        im = cv2.imread(name, color_mode)

        # search for normalization factor (max no. to divide by)
        max_no = np.iinfo(im.dtype).max

        if im.dtype != dtype:
            im = np.asarray(im, dtype=dtype)

        if max_no and im.dtype in [np.float16, np.float32, np.float64]:
            im = im / max_no

        images.append(im)

    return images

def propagateAngularSpectrum(z, ui, wl, n, dx):
    """Propagate optical field given in angular spectrum (ui) onto given distance z.
    :param z            propagation distance
    :param ui           numpy array-like in space domain
    :param n            refractive index
    :param dx           sample size
    :param wl           wavelength
    :return             numpy array with output field after propagation
    """
    # Convertion to spacial frequency coordinates
    Nx, Ny = ui.shape
    dfx, dfy = 1/Nx/dx, 1/Ny/dx
    fx = np.arange(-Nx//2, Nx//2) * dfx
    fy = np.arange(-Ny//2, Ny//2) * dfy
    FX, FY = np.meshgrid(fy, fx)

    # 2D DFT of the ui and shift 0-freq. component to the center of the spectrum
    ui_FT = np.fft.fftshift(np.fft.fft2(ui))

    # Transfer Function - rigourous (Fresnel) approach
    # 0j is explicitly given to np.sqrt behave correctly
    fz = np.sqrt((n/wl)**2 + 0j - np.square(FX) - np.square(FY))
    TF = np.exp(2j*np.pi*fz*z)

    # FT of convolution: fft(A(*)B) = fft(A)*fft(B)
    uo_FT = np.multiply(ui_FT, TF)

    return np.fft.ifft2(np.fft.ifftshift(uo_FT))

def fiveFrameTemporalPhaseShifting(frames):
    """perform 5-frame TPS
    :param frames   list of 5 phase frames (numpy 2d arrays)
    :return         tuple restored phase of the img (numpy 2d arrays)
    """
    for fr in frames:
        assert type(fr) == np.ndarray, "argument should be list of numpy arrays"

    re = 2*(frames[1] - frames[3])
    im = 2*frames[2] - frames[0] - frames[4]
    phase = np.abs(np.arctan2(re, im))  # like matlab atan2: result is [-pi, pi]
    amp = np.sqrt(np.square(re) + np.square(im))

    return phase, amp

