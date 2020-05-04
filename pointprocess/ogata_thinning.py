import random

from pointprocess.parameters import HawkesParams


class OgataThinning:
    def _simulate_ogata_thinning_till_time(self, params: HawkesParams, T):
        # is_done = lambda time, T: time >= T
        current_time = 0
        current_sample_idx = 0
        history = []
        while current_time < T:
            current_lambda = self.compute_intensity(current_time, history)
            candidate_interval = random.expovariate(current_lambda)
            current_time += candidate_interval

            candidate_lambda = self.compute_intensity(current_time, history)
            if random.random() < (candidate_lambda / current_lambda):
                current_sample_idx += 1
                history.append(current_time)

        return history if history[-1] < T else history[:-1]

    def _simulate_ogata_thinning_fixed_total_events(self, params: HawkesParams, N):
        # is_done = lambda event_id, N: event_id >= N
        current_time = 0
        current_sample_idx = 0
        history = []
        while current_sample_idx < N:
            current_lambda = self.compute_intensity(current_time, history)
            candidate_interval = random.expovariate(current_lambda)
            current_time += candidate_interval

            candidate_lambda = self.compute_intensity(current_time, history)
            if random.random() < (candidate_lambda / current_lambda):
                current_sample_idx += 1
                history.append(current_time)

        return history

    def simulate_ogata_thinning(self, params: HawkesParams, T=None, N=None):
        if T is not None:
            return self._simulate_ogata_thinning_till_time(params, T)
        else:
            return self._simulate_ogata_thinning_fixed_total_events(params, N)