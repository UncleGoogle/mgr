import sys
import cv2
import argparse
import datetime
import json

import numpy as np
from matplotlib import pyplot as plt

from img import ExperimentalImg, SimulationImg, ImgSet
from slider import cv2_slider


@cv2_slider(ksize=7)
def gradient_sobel(im, ksize, slider=True):
    """Computes x and y sobel gradient function
    Usage: imp, _ = gradient_sobel(imp, ksize=default_int, slider=bool)
    _
    :param im       image - required by slider decorator
    :param ksize    int - initial value for slider
    :param slider   wheather use slider or just compute once
    """
    if ksize < 1:
        ksize = 1
    elif ksize % 2 == 0:
        ksize = ksize - 1
    sobelx64f = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u, ksize


# -------------------------------------Parsing script argments----------------
parser = argparse.ArgumentParser()
parser.add_argument('expJson', type=str, help="file describing experimental data in JSON format")
parser.add_argument('experimentalDir', type=str, help="directory with experimental images")
parser.add_argument('mappingJson', type=str, help="file having simulation results as linear eq params")
parser.add_argument('--output', type=str, help="where to store results TODO")
args = parser.parse_args()

# -------------------------------------Settings-------------------------------
px_camera_size = 0.001 * 4.29  # EOS 650D sensor pixel size in mm
row = "{:<28}\t{:>6.0f}px\t{:>5.3f}mm"
log_name = datetime.datetime.now()

with open(args.mappingJson, 'r') as out_file:
    mapping = json.load(out_file)
    m = mapping['m']
    b = mapping['b']

# -------------------------------------reading data---------------------------
experimental_data = ImgSet.load_experimental_dataset(args.expJson, args.experimentalDir)

print("Datasets found in given json:")
while True:
    for key in experimental_data.keys():
        print('->', key)
    chosen_option = input('Choose dataset: ')
    try:
        dataset = experimental_data[chosen_option]
    except KeyError:
        print(f'No data {chosen_option} found. Try again', end="")
    else:
        break

pixel_size = dataset.pixel_size
if pixel_size is None:
    print('WARNING: No pixel size defined. Uses default Canon 650D: 4,29um')
    pixel_size = px_camera_size

x_es, a_ses = [], []

for img in dataset:
    if img.kind != 'dot':
        continue

    # -------------------------------------looking for spot diameter--------------
    img.load_image()

    img.im = cv2.normalize(img.im, img.im, 0, 255, norm_type=cv2.NORM_MINMAX)
    scale = 1
    img.resize_im(scale_down_by=scale)
    a_px = img.read_a()  # by fitting circle
    if not a_px or a_px == -1:
        # No circle fitted. Skip.
        continue

    a = a_px * scale * pixel_size
    print(row.format(f'{img}', a_px, a))

    # -------------------------------------write output---------------------------
    x_es.append(img.x)
    a_ses.append(a)
    with open(f'./output_compare/{log_name}', 'a+') as out_log:
        out_log.write(row.format(f'{img}', a_px, a))
        out_log.write('\n')

# -------------------------------------put on a plot-------------------------
run = True
print([m*x+b for x in x_es])
print(x_es)
while run:
    magic_no = float(input('new magic no:') or 1)

    plt.plot(x_es, [a*magic_no for a in a_ses], 'bx--',
            label=f'read from experiment [x {magic_no}]', linewidth=0, markersize=6)
    plt.plot(x_es, [m*x+b for x in x_es], 'g--',
            label='Mapping. Theoretical spot sizes.\ny={:.4f}x + {:.4f}'.format(m, b))
    plt.title('Matching displacement with the mapping.')
    plt.xlabel('distance from focal point, x [mm]')
    plt.ylabel('average spot diameter, a [mm]')
    plt.legend()
    plt.show()

