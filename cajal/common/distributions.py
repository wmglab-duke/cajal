import numpy as np
import scipy
from scipy import spatial

from cajal.mpi.__backend import Backend as MPISRC
from cajal.mpi.random import RNG


class Distribution:
    def __init__(self, ndim, random_state=None):
        self.rng = random_state or RNG
        self.ndim = ndim

    def pull(self, samples):
        raise NotImplementedError()


class Sampler(Distribution):
    def __init__(self, bounds, random_state=None):
        super().__init__(len(bounds), random_state)
        self.limits = np.asfarray(bounds).T
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

    def unitpop(self, samples):
        raise NotImplementedError()

    def _unitpop(self, n):
        if MPISRC.ENABLED:
            if MPISRC.MASTER():
                samples = np.asfarray(self.unitpop(n), dtype="f")
            else:
                samples = np.empty((n, self.ndim), dtype="f")
            MPISRC.COMM.Bcast(samples, root=MPISRC.MASTER_RANK)
            return samples
        return self.unitpop(n)

    def pull(self, samples):
        return self.__scale_arg1 + (self._unitpop(samples) - 0.5) * self.__scale_arg2

    def sync_pull(self, samples):
        return self.__scale_arg1 + (self.unitpop(samples) - 0.5) * self.__scale_arg2


class LatinHypercubeSampler(Sampler):
    def unitpop(self, samples):
        segsize = 1.0 / samples
        samples_arr = (
            segsize * self.rng.random((samples, self.ndim))
            + np.linspace(0.0, 1.0, samples, endpoint=False)[:, np.newaxis]
        )
        pop = np.zeros_like(samples_arr, dtype="f")
        for j in range(self.ndim):
            order = self.rng.permutation(range(samples))
            pop[:, j] = samples_arr[order, j]

        return pop


class UniformRandomSampler(Sampler):
    def unitpop(self, samples):
        pop = self.rng.random((samples, self.ndim))
        return pop


class NormalSampler(Sampler):
    def unitpop(self, samples):
        pop1 = self.rng.random((samples, self.ndim))
        pop2 = self.rng.random((samples, self.ndim))
        return (pop1 + pop2) / 2


class ChaoticSampler(Sampler):
    def __init__(self, bounds, iters, random_state=None):
        self.iters = iters
        super().__init__(bounds, random_state)

    def unitpop(self, samples):
        pop = self.rng.random((samples, self.ndim))
        for i in range(self.iters):
            pop[:] = np.sin(np.pi * pop)
        return pop


class SLHSampler(Sampler):
    def __init__(self, bounds, slices, maxeval=50, random_state=None):
        self.slices = slices
        self.maxeval = maxeval
        super().__init__(bounds, random_state)

    @staticmethod
    def minimum_distance(X):
        return np.min(spatial.cKDTree(X).query(X, 2)[0][:, 1])

    @staticmethod
    def mean_min_dist(X, t):
        return np.mean(
            [
                SLHSampler.minimum_distance(X[i * t : i * t + t])
                for i in range(int(X.shape[0] / t))
            ]
        )

    @staticmethod
    def LHD_cost(X):
        n, d = X.shape
        edges = np.linspace(0, 1, n)
        Y = np.digitize(X.T, edges)
        totals = [np.sum(np.histogram(np.unique(i), range(1, n + 1))[0]) for i in Y]
        return -np.sum(totals)

    @staticmethod
    def rand_perm_interval(lb, ub, t):
        perm = np.random.permutation(np.arange(lb, ub + 1))
        return perm[:t]

    def generate_sample(self, n, m):
        # n => number of sample points
        # p => number of parameters
        # t => number of slices / sub-samples
        # m => size of slice
        p = self.ndim
        t = self.slices
        y = np.empty((p, n), dtype=int)

        for i in range(p):
            y[i, :] = np.concatenate(
                [np.random.permutation(np.arange(m)) for _ in range(t)]
            )
        yc = y.copy()

        for j in range(p):
            for k in range(m):
                lb = k * t + 1
                ub = (k + 1) * t
                pp = self.rand_perm_interval(lb, ub, t)
                y[j, np.where(yc[j, :] == k)[0]] = pp

        y = y.T
        y = np.random.uniform(y - 1, y)
        y /= n

        return y

    def unitpop(self, samples):
        # n => number of sample points
        # d => number of parameters
        # t => number of slices/sub-samples (n = m * t)
        n = samples
        t = self.slices
        m = int(int(n) / int(t))

        best_sample = self.generate_sample(n, m)
        best_sample_cost = self.minimum_distance(best_sample)
        best_sub_sample_cost = self.mean_min_dist(best_sample, t)
        cost = (best_sample_cost + best_sub_sample_cost) / 2

        for _ in range(1, self.maxeval):
            new_sample = self.generate_sample(n, m)
            new_sample_cost = self.minimum_distance(new_sample)
            new_sub_sample_cost = self.mean_min_dist(new_sample, t)
            new_cost = (new_sample_cost + new_sub_sample_cost) / 2

            if new_cost > cost:
                best_sample = new_sample
                cost = new_cost

        return best_sample


class SobolSampler(Sampler):
    def __init__(self, bounds, random_state=None):
        super().__init__(bounds, random_state)
        self.sob = scipy.stats.qmc.Sobol(self.ndim, seed=self.rng)

    def unitpop(self, samples):
        return self.sob.random(samples)
