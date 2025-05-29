import numpy as np
from numba import njit


@njit
def wrap_heading(heading):
    """Ensures input heading is between -180 an 180; can be float or np.ndarray"""
    return (heading + np.pi) % (2 * np.pi) - np.pi


def handle_single_point(func):
    def wrapper(coords):
        # Check if coords is a single point (1D array) or multiple points (2D array)
        if len(coords.shape) == 1:
            # If it's a single point, add an extra dimension to make it a 2D array
            coords = np.expand_dims(coords, axis=0)
            # Call the original function and remove the extra dimension from the result
            return np.squeeze(func(coords))
        else:
            # If multiple points, just call the original function
            return func(coords)

    return wrapper


@handle_single_point
def habitat_to_xyz(coords: np.ndarray) -> np.ndarray:
    return np.array([-coords[:, 2], -coords[:, 0], coords[:, 1]]).T


@handle_single_point
def xyz_to_habitat(coords: np.ndarray) -> np.ndarray:
    return np.array([-coords[:, 1], coords[:, 2], -coords[:, 0]]).T
