import sys
import cv2
import argparse
import pathlib
import os
import numpy as np

from helpers.readImage import readImage
from img import ExperimentalImg, SimulationImg, ImgSet

from slider import cv2_slider


thresh_max = 255
method_max = 16
@cv2_slider(method=method_max, threshold=thresh_max)
def binary_threshold(im, method, threshold):
    ret, res = cv2.threshold(im, threshold, thresh_max, method)
    return res, ret

@cv2_slider(maxVal=50, minVal=50)
def canny_edge(im, maxVal, minVal):
    res = cv2.Canny(img.im, minVal, maxVal)
    return res

@cv2_slider(ksize=7)
def gradient_sobel(im, ksize):
    if ksize < 1:
        ksize = 1
    elif ksize % 2 == 0:
        ksize = ksize - 1
    sobelx64f = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u, ksize

@cv2_slider(ksize=31)
def gradient_laplacian(im, ksize):
    ksize += (ksize + 1) % 2
    lapl = cv2.Laplacian(im, cv2.CV_64F, ksize=ksize)
    abs = np.absolute(lapl)
    lapl_8u = np.uint8(abs)
    return lapl_8u, ksize

@cv2_slider(ksize=11)
def opening(im, ksize):
    ksize += (ksize + 1) % 2
    kernel = np.ones((ksize,ksize), np.uint8)
    return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)


def main():
    # -------------------------------------Parsing script argments----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('expJson', type=str, help="file describing experimental data in JSON format")
    parser.add_argument('experimentalDir', type=str, help="directory with experimental images")
    parser.add_argument('simulationDir', type=str, help="directory with simulation images")
    parser.add_argument('--output', type=str, help="where to store results")
    args = parser.parse_args()

    # -------------------------------------Settings-------------------------------
    # in milimeters
    wl = 0.00628
    f = 50
    r = 27.75
    d = 2*r
    sampling = 0.001 * 1  # mm to px ratio
    px_camera_size = 0.001 * 4.29  # EOS 650D sensor pixel size in mm

    pre_propostions_a  = r / f
    row = "{:<28} {:>6.0f}px {:>5.3f}mm"

    # -------------------------------------reading data---------------------------
    simulation_data = ImgSet.load_simulation_dataset(args.simulationDir, "Airy simulation")
    experimental_data = ImgSet.load_experimental_dataset(args.expJson, args.experimentalDir)

    # -------------------------------------chosing x for investigation------------
    x = 26  # mm
    sim = simulation_data.get_dot_for_x(x)
    img = experimental_data['maxwell'].get_dot_for_x(x)  # maxwell is just dataset name

    # -------------------------------------CoC size from proportions--------------
    proportions_a = r / f * x
    print(row.format('CoC (proportionally)', 0.0, proportions_a))

    # -------------------------------------loading simulations--------------------
    sim.load_image()

    threshold = 8  # simple cutoff for pixels for simulation
    a_px = sim.read_a(threshold=threshold)
    a = a_px * sampling
    print(row.format(f'{sim}', a_px, a))

    # -------------------------------------loading and processing experimental----
    img.load_image()
    scale = 2
    img.resize_im(scale_down_by=scale)

    # interactive sliders to choose proper parameters -- see saved images
    img.im = cv2.medianBlur(img.im, ksize=5)
    img.im, _ = binary_threshold(img.im, method=cv2.THRESH_BINARY, threshold=10)
    img.im = opening(img.im, ksize=5)
    img.im, _ = gradient_sobel(img.im, ksize=7)

    a_px = img.read_a()
    a = a_px * scale * px_camera_size
    print(row.format(f'{img}', a_px, a))

    # -------------------------------------processing exp data--------------------

#     data = [img for img in experimental_data['maxwell'] if img.kind=='dot']
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

    #     # -----------------------------Measure a from an image-------------------
    #     a_px = read_a(im, threshold=threshold)
    #     print(row.format(f'Simulation (treshhold={threshold})', a_px * sampling, a_px))
    #     print(row.format('CoC to simulation:', proportions_a_px/a_px, 0))
    #     # cv2.circle(im, (im.shape[0]//2, im.shape[1]//2-1), int(a_px//2), (255, 0, 0))
    #     # cv2.imshow('image', im)
    #     # cv2.waitKey(0)

if __name__ == '__main__':
    main()
