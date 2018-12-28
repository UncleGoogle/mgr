import argparse
import datetime
import json

import numpy as np
from matplotlib import pyplot as plt
import cv2

from img import SimulationImg, ImgSet

parser = argparse.ArgumentParser()
parser.add_argument('simulationDir', type=str, help="directory with simulation images")
parser.add_argument('threshold', type=int, help="[0-255] cutoff for diameter read")
parser.add_argument('--output', type=str, help="where to store results")
args = parser.parse_args()

threshold = args.threshold
sampling = 0.001 * 1  # mm to px ratio

f = 250
r = 27.75
d = 2*r

simulation_data = ImgSet.load_simulation_dataset(args.simulationDir, "Airy simulation")

mapping_sim = {}
mapping_prop = {}
for sim in simulation_data:
    sim.load_image()
    a_px_sim = sim.read_a(threshold=threshold)
    a_sim = a_px_sim * sampling
    mapping_sim[sim.x] = a_sim

x_sim, y_sim = zip(*sorted(mapping_sim.items()))
x_sim = np.array(x_sim)

# -------------------------------------fitting line---------------------------
# m*x+b = y  =>  y = Ap, whre p=[[m], [b]], A=[[x 1]]
A = np.vstack([x_sim, np.ones(len(x_sim))]).T
m, b = np.linalg.lstsq(A, y_sim)[0]

# -------------------------------------put on plot----------------------------
x_mock = np.array([0, x_sim[-1]])
plt.plot(x_mock, d/f*x_mock, 'g-', label=f'Incoherent light (CoC proportionally: a=r*x/f)', linewidth=1, markersize=0)
plt.plot(x_sim, y_sim, 'rx--', label=f'Coherent light - simulations (threshold: {threshold}/255)', linewidth=0, markersize=6)
# without intercept - dont use b below
plt.plot(x_sim, m*x_sim, 'r--', label='Fitted line(simulations)')
plt.title('CoC size along displacement for bigger lens')
plt.xlabel('distance from focal point, x [mm]')
plt.ylabel('spot diameter, a [mm]')
plt.legend()
plt.show()

log_time = datetime.datetime.now()

output_json = {'m': m, 'b': b, 'threshold': threshold, 'experimental_data': mapping_sim, 'generated': str(log_time)}
with open(f'./output_mapping/mapping.json', 'w') as out_file:
    json.dump(output_json, out_file, indent=4)

