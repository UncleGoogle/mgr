import cv2
import numpy as np

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

