import argparse

import numpy as np
from matplotlib import pyplot as plt

from img import SimulationImg, ImgSet

parser = argparse.ArgumentParser()
parser.add_argument('simulationDir', type=str, help="directory with simulation images")
parser.add_argument('threshold', type=int, help="[0-255] cutoff for diameter read")
parser.add_argument('--output', type=str, help="where to store results")
args = parser.parse_args()

threshold = args.threshold
threshold_relative = round(threshold / 255, 2)
sampling = 0.001 * 1  # mm to px ratio

f = 50
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
# plt.plot(x_prop, y_prop, 'gx--', label='Incoherent light (CoC proportionally: a=r*x/f)', linewidth=0, markersize=6)
x_mock = np.array([0, x_sim[-1]])
plt.plot(x_mock, r/f*x_mock, 'g-', label='Incoherent light (CoC proportionally: a=r*x/f)', linewidth=1, markersize=0)
plt.plot(x_sim, y_sim, 'rx--', label=f'Coherent light - simulations (threshold: {threshold_relative})', linewidth=0, markersize=6)
plt.plot(x_sim, float(m)*x_sim + b, 'r--', label='Fitted line(simulations)')
plt.title('CoC radius size along displacement for bigger lens')
plt.xlabel('distance from focal point, x [mm]')
plt.ylabel('spot radius, a [mm]')
plt.legend()
plt.show()

with open(f'./output_mapping/mapping', 'w') as out_file:
    out_file.write(mapping_sim)

