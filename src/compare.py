import cv2
import argparse
import datetime

import numpy as np

from img import ExperimentalImg, SimulationImg, ImgSet
from slider import cv2_slider


@cv2_slider(ksize=7)
def gradient_sobel(im, ksize, slider=True):
    if ksize < 1:
        ksize = 1
    elif ksize % 2 == 0:
        ksize = ksize - 1
    sobelx64f = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u, ksize

@cv2_slider(ksize=31)
def gradient_laplacian(im, ksize, slider=True):
    ksize += (ksize + 1) % 2
    lapl = cv2.Laplacian(im, cv2.CV_64F, ksize=ksize)
    abs = np.absolute(lapl)
    lapl_8u = np.uint8(abs)
    return lapl_8u, ksize

# sliders usage
# imp, _ = gradient_sobel(imp, ksize=3, slider=True)


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
    f = 250
    r = 27.75
    d = 2*r
    sampling = 0.001 * 1  # mm to px ratio
    px_camera_size = 0.001 * 4.29  # EOS 650D sensor pixel size in mm

    pre_propostions_a  = r / f
    row = "{:<28}\t{:>6.0f}px\t{:>5.3f}mm"

    log_name = datetime.datetime.now()

    # -------------------------------------reading data---------------------------
    simulation_data = ImgSet.load_simulation_dataset(args.simulationDir, "Airy simulation")
    experimental_data = ImgSet.load_experimental_dataset(args.expJson, args.experimentalDir)

    exp_results = []
    sim_results = []
    for img in experimental_data['maxwell']:
        if img.kind != 'dot':
            continue

        # -------------------------------------chosing x for investigation------------
        try:
            sim = simulation_data.get_dot_for_x(img.x)
        except IndexError:
            around = int(img.x * 0.1)
            smallest_diff = 100000
            for s in simulation_data:
                if s.kind != 'dot':
                    continue
                diff = abs(s.x - img.x)
                if diff <= around and diff < smallest_diff:
                    smallest_diff = diff
                    sim = s
            print(f'No simulation data for +/- {around} around experimental distance {img.x}. Skipping...')
            continue

        print(f'experiment: {sim.x}, simulation: {sim.x}')

        # -------------------------------------CoC size from proportions--------------
        proportions_a = d / f * img.x

        # -------------------------------------loading simulations--------------------
        sim.load_image()

        threshold = 8  # simple cutoff for pixels for simulation
        a_px_sim = sim.read_a(threshold=threshold)
        a_sim = a_px_sim * sampling

        # -------------------------------------loading and processing experimental----
        img.load_image()
        scale = 2
        img.resize_im(scale_down_by=scale)

        # read diameter of the spot by fitting circle
        a_px = img.read_a()
        a = a_px * scale * px_camera_size

        # -------------------------------------write output---------------------------

        exp_results.append(a_sim)
        sim_results.append(a)

        print(row.format('CoC (proportionally)', 0.0, proportions_a))
        print(row.format(f'{sim}', a_px_sim, a_sim))
        print(row.format(f'{img}', a_px, a))

        with open(f'./output_compare/{log_name}', 'a+') as out_log:
            out_log.write(row.format('CoC (proportionally)', 0.0, proportions_a))
            out_log.write('\n')
            out_log.write(row.format(f'{sim}', a_px_sim, a_sim))
            out_log.write('\n')
            out_log.write(row.format(f'{img}', a_px, a))
            out_log.write('\n\n')

    # -------------------------------------put on a plot-------------------------
    x_mock = np.array([0, x_sim[-1]])
    plt.plot(x_mock, d/f*x_mock, 'g-', label=f'Incoherent light (CoC proportionally: a=r*x/f)', linewidth=1, markersize=0)
    plt.plot(x_sim, y_sim, 'rx--', label=f'Coherent light - simulations (threshold: {threshold}/255)', linewidth=0, markersize=6)
    plt.plot(x_sim, float(m)*x_sim + b, 'r--', label='Fitted line(simulations)')
    plt.title('CoC radius size along displacement for bigger lens')
    plt.xlabel('distance from focal point, x [mm]')
    plt.ylabel('spot radius, a [mm]')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
