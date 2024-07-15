import math
import numpy as np

from cajal.mpi.random import RNG


__all__ = ("rand_fibre_locations", "sunflower")


def rand_fibre_locations(N_axons, nerve_radius, N_configs=1, with_polar=False):
    """Give nerve radius in um"""

    # np.random.seed(seed=1)  # set random num gen seed for testing purposes

    theta = 2 * np.pi * RNG.rand(N_configs, N_axons)
    R = nerve_radius * np.sqrt(RNG.random(N_configs, N_axons))
    x = np.multiply(R, np.cos(theta))
    y = np.zeros_like(x)
    z = np.multiply(R, np.sin(theta))

    if with_polar:
        fibre_locations = np.stack((x, y, z, R, theta), axis=2)
    else:
        fibre_locations = np.stack((x, y, z), axis=2)

    return fibre_locations


def sunflower(N_axons, nerve_radius, alpha=2):
    """Generate sunflower axon arrangement.

    Parameters
    ----------
    N_axons : int
        Number of axons
    nerve_radius : float
        nerve radius in um
    alpha : int, optional
        Boundary smoothing factor, by default 2

    Returns
    -------
    array
        Fibre locations.
    """

    # number of boundary point
    b = round(alpha * math.sqrt(N_axons))

    phi = (math.sqrt(5) + 1) / 2
    fibre_locations = np.empty([N_axons, 2])
    for i in range(0, N_axons):
        r = radius(i + 1, N_axons, b, nerve_radius)
        theta = 2 * math.pi * i / phi**2
        fibre_locations[i, 0] = r * math.cos(theta)
        fibre_locations[i, 1] = r * math.sin(theta)
    return fibre_locations


def radius(i, N_axons, b, nerve_radius):
    """radial smoothing for sunflower function."""
    if i > N_axons - b:
        r = nerve_radius
    else:
        r = nerve_radius * math.sqrt(i - 1 / 2) / math.sqrt(N_axons - (b + 1) / 2)
    return r
