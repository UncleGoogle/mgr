import sys
import cv2
import argparse
import pathlib
import os
import numpy

from helpers.readImage import readImage
from img import Img, ImgSet

def main():
    # -------------------------------------Arguments & Settings-------------------
    # print whole array
    numpy.set_printoptions(threshold=sys.maxsize)

    # in milimeters
    wl = 0.00628
    f = 50
    r = 27.75
    d = 2*r
    sampling_ls = 1  # um to px ratio
    sampling = 0.001 * sampling_ls  # mm to px ratio
    threshold = 8  # cutoff for pixels

    pre_propostions_a  = r / f
    pre_airy_a = 1.22 * wl / d

    parser = argparse.ArgumentParser()
    parser.add_argument('inputDir', type=str, help="directory with files to be processed")
    parser.add_argument('--output', type=str, help="where to store results")
    parser.add_argument('--data_json', type=str, help="data info file")
    parser.add_argument('--exp', type=str, help="directory with experimental data to be processed")
    args = parser.parse_args()

    # -------------------------------------reading experimental data--------------

    experimental_data = ImgSet.load_from_json_factory(args.data_json, args.exp)
    # import ipdb; ipdb.set_trace();
    data = filter(lambda img: img.kind == 'dot', experimental_data['maxwell'])
    scale = 4
    row = "{:<28} {:>5.3f}mm {:>6.0f}px"
    for img in data:
        img.load_image()
        img.resize_im(scale_down_by=scale)

        # -----------------------------Measure diameter from an image------------
        a_px = img.read_a() * scale
        print(row.format(f'{img.kind} at {img.x}:', a_px * sampling, a_px))

        # -----------------------------Calculate CoC proportional sizes----------
        proportions_a = pre_propostions_a * img.x
        proportions_a_px = proportions_a // sampling
        print(row.format('CoC (proportionally)', proportions_a, proportions_a_px))


        # print(row.format('CoC to simulation:', proportions_a_px/a_px, 0))
        # cv2.circle(im, (im.shape[0]//2, im.shape[1]//2-1), int(a_px//2), (255, 0, 0))

        # print(f'Img {img} ---> read_a: {diameter}')

    #     # commented out becase airy middle disk investigation is depracated
    #     # airy_a = pre_airy_a * (f + x)
    #     # airy_a_px = airy_a // sampling
    #     # print(row.format('Airy first disk', airy_a, airy_a_px))

    #     # -----------------------------Measure a from an image-------------------
    #     a_px = read_a(im, threshold=threshold)
    #     print(row.format(f'Simulation (treshhold={threshold})', a_px * sampling, a_px))
    #     print(row.format('CoC to simulation:', proportions_a_px/a_px, 0))
    #     # cv2.circle(im, (im.shape[0]//2, im.shape[1]//2-1), int(a_px//2), (255, 0, 0))
    #     # cv2.imshow('image', im)
    #     # cv2.waitKey(0)

        # tries = 1
        # while tries < 5:
        #     diameter = img.read_a() * (scale ** tries)
        #     if diameter > 0:
        #         print(f'radius is {diameter} (image was resized {scale ** tries} times)')
        #         break
        #     print('No circle found. Scaling down by {}')
        #     tries += 1
        #     img.resize(scale_down_by=scale)


    # img1 = data[0][0]
    # img1.readImage()
    # img1.resize_im(divide_by=8)
    # print('circle diameter is: ', 8 * img1.read_a())

    # -------------------------------------reading images-------------------------

    # for p in pathlib.Path(args.inputDir).iterdir():
    #     if not (p.is_file() and p.suffix == '.bmp' or p.suffix == ".JPG"):
    #         continue

    #     # im = cv2.imread(str(p.absolute()), 0)
    #     im = readImage(str(p.absolute()))

    #     # -----------------------------Reading x from image filename-------------
    #     x = int(p.stem.split('_')[-1])
    #     print(f'------- x = {x} -------')

    #     # -----------------------------Calculate theoretical sizes---------------
    #     row = "{:<28} {:>5.3f}mm {:>6.0f}px"
    #     proportions_a = pre_propostions_a * x
    #     proportions_a_px = proportions_a // sampling
    #     print(row.format('CoC (proportionally)', proportions_a, proportions_a_px))

    #     # commented out becase airy middle disk investigation is depracated
    #     # airy_a = pre_airy_a * (f + x)
    #     # airy_a_px = airy_a // sampling
    #     # print(row.format('Airy first disk', airy_a, airy_a_px))

    #     # -----------------------------Measure a from an image-------------------
    #     a_px = read_a(im, threshold=threshold)
    #     print(row.format(f'Simulation (treshhold={threshold})', a_px * sampling, a_px))
    #     print(row.format('CoC to simulation:', proportions_a_px/a_px, 0))
    #     # cv2.circle(im, (im.shape[0]//2, im.shape[1]//2-1), int(a_px//2), (255, 0, 0))
    #     # cv2.imshow('image', im)
    #     # cv2.waitKey(0)

if __name__ == '__main__':
    main()
