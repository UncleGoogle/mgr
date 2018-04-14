import sys
import argparse

import cv2
import matplotlib.pyplot as plt

from helpers.readImage import readImage


parser = argparse.ArgumentParser()
parser.add_argument('image')
parser.add_argument('--t1', type=int, help='threshold1')
parser.add_argument('--t2', type=int, help='threshlod2 > threshold1')
args = parser.parse_args()

# -------------------------------------reading images-------------------------

im = readImage(args.image, color_mode=0)  # 0=grayscale

# -------------------------------------thresholding---------------------------

# threshold, res = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
_, res = cv2.threshold(im, args.t1, 255, cv2.THRESH_TOZERO)
_, res = cv2.threshold(res, args.t2, 255, cv2.THRESH_TRUNC)


images = [im, res]
titles = ['input image', 'binary threshold']

for i in range(2):
    plt.subplot(1,2,1+i)
    plt.title(titles[i])
    plt.imshow(images[i])

plt.show()
