import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pointprocess.parameters import HawkesParams
from pointprocess.ogata_thinning import OgataThinning


class PlainHawkes(OgataThinning):
    def __init__(self, params: HawkesParams):
        self.params = params
        self.kernel = lambda x, alpha, beta: alpha * np.exp(-beta * x)

    def simulate(self, num_events_per_sequence, num_sequence):
        sequences = [self.simulate_ogata_thinning(self.params, N=num_events_per_sequence) for _ in range(num_sequence)]
        return sequences

    def compute_intensity(self, query_time: float, history: list):
        history = [t for t in history if t < query_time]
        return self.params.mu + np.sum(self.kernel(query_time - np.array(history), self.params.alpha, self.params.beta))

    def plot(self, sequence, ax: Axes):
        T = sequence[-1]
        grid_times = np.linspace(0, T, 1000)
        grid_intensities = [self.compute_intensity(_t, sequence) for _t in grid_times]
        ax.plot(grid_times, grid_intensities)
        ax.scatter(sequence, np.zeros_like(sequence))


def plot(model, simulated_points):
    fig, ax = plt.subplots(1, 1)
    model.plot(simulated_points, ax)
    plt.show()


def fast_simulating_standard_hawkes():
    params = HawkesParams(0.2, 0.8, 1)
    hawkes = PlainHawkes(params)
    num_events = 20

    start_time = time.time()  # seconds
    sequences = hawkes.simulate(num_events, 1)
    end_time = time.time()  # seconds
    print(f"Simulating {num_events} events took: {end_time - start_time:.3f} seconds")
    plot(hawkes, sequences[0])


if __name__ == '__main__':
    fast_simulating_standard_hawkes()
