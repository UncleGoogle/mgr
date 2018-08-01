import argparse
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
    # CoC from simple proportions:
    mapping_prop[sim.x] = r / f * sim.x

x_sim, y_sim = zip(*sorted(mapping_sim.items()))
x_prop, y_prop = zip(*sorted(mapping_prop.items()))

plt.plot(x_sim, y_sim, 'go--', label=f'Coherent light - simulations (threshold: {threshold_relative})', linewidth=0, markersize=6)
plt.plot(x_prop, y_prop, 'gx--', label='Incoherent light (CoC proportionally: a=r*x/f)', linewidth=0, markersize=6)
plt.title('CoC radius size along displacement for bigger lens')
plt.xlabel('distance from focal point, x [mm]')
plt.ylabel('spot radius, a [mm]')
plt.legend()
plt.show()

