import sys
import cv2
import argparse
import pathlib
import os
import numpy as np

from helpers.readImage import readImage
from img import ExperimentalImg, SimulationImg, ImgSet

from slider import cv2_slider


def main():
    # -------------------------------------Arguments & Settings-------------------
    # print whole array
    np.set_printoptions(threshold=sys.maxsize)

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

    row = "{:<28} {:>5.3f}mm {:>6.0f}px"

    parser = argparse.ArgumentParser()
    parser.add_argument('expJson', type=str, help="file describing experimental data in JSON format")
    parser.add_argument('experimentalDir', type=str, help="directory with experimental images")
    parser.add_argument('simulationDir', type=str, help="directory with simulation images")
    parser.add_argument('--output', type=str, help="where to store results")
    args = parser.parse_args()

    # -------------------------------------reading simulation data----------------
    # simulation_data = ImgSet.load_simulation_dataset(args.simulationDir, "Airy simulation")
    # for sim in simulation_data:
    #     sim.load_image()
    #     a_px = sim.read_a(threshold=threshold)
    #     a = a_px * sampling
    #     print(row.format(f'{sim}', a, a_px))

    # -------------------------------------reading experimental data--------------

    experimental_data = ImgSet.load_experimental_dataset(args.expJson, args.experimentalDir)
    data = filter(lambda img: img.kind == 'dot', experimental_data['maxwell'])
    scale = 2

    win = 'preview'

    img = list(data)[3]  # 6mm from the focal point
    img.load_image()
    img.resize_im(scale_down_by=scale)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    @cv2_slider('slider', ksize=7)
    def gradient(im, ksize):
        if ksize < 1:
            ksize = 1
        elif ksize > 7:
            ksize = 7
        elif ksize % 2 == 0:
            ksize = ksize -1
        sobelx64f = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=ksize)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_8u = np.uint8(abs_sobel64f)
        return ksize, sobel_8u

    @cv2_slider('slider', ksize=31)
    def gradient_laplacian(im, ksize):
        if ksize < 1:
            ksize = 1
        elif ksize > 31:
            ksize = 31
        elif ksize % 2 == 0:
            ksize = ksize -1
        lapl = cv2.Laplacian(im, cv2.CV_64F, ksize=ksize)
        abs = np.absolute(lapl)
        lapl_8u = np.uint8(abs)
        return ksize, lapl_8u

    chosen_ksize, res = gradient_laplacian(img.im, ksize=3)

    cv2.imshow(win, res)
    cv2.waitKey()

#     for img in data:
#         img.load_image()
#         img.resize_im(scale_down_by=scale)

#         # -----------------------------Measure diameter from an image------------
#         a_px = img.read_a() * scale
#         print(row.format(f'{img.kind} at {img.x}:', a_px * sampling, a_px))

#         # -----------------------------Calculate CoC proportional sizes----------
#         proportions_a = pre_propostions_a * img.x
#         proportions_a_px = proportions_a // sampling
#         print(row.format('CoC (proportionally)', proportions_a, proportions_a_px))


















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
