import numpy as np


def circumferential_locations(N_electrodes, electrode_distance, rotation=0):
    """Get x,y,z coordinates of electrodes arranged in an evenly spaced
    ring electrode_distance from the origin"""

    theta = (
        2
        * np.pi
        * np.linspace(0.0 + rotation, 1.0 + rotation, num=N_electrodes, endpoint=False)
    )
    x = electrode_distance * np.cos(theta)
    y = np.zeros(N_electrodes)
    z = electrode_distance * np.sin(theta)

    electrode_locations = np.stack((x, y, z), axis=1)
    return electrode_locations
