"""
 utils.py

 Helper functions that are used in multiple files. Feel free to add more functions.

 Dynamic Programming and Optimal Control
 Fall 2024
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch

 Authors: Maximilian Stralz, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from Constants import Constants

def bresenham(start, end):
    """
    Generates the coordinates of a line between two points using Bresenham's algorithm.

    Parameters:
        start (tuple or list): The starting point (x0, y0).
        end (tuple or list): The ending point (x1, y1).

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates.

    Example:
        >>> bresenham((2, 3), (10, 8))
        [(2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 6), (8, 7), (9, 7), (10, 8)]
    """
    x0, y0 = start
    x1, y1 = end

    points = []

    dx = x1 - x0
    dy = y1 - y0

    x_sign = 1 if dx > 0 else -1 if dx < 0 else 0
    y_sign = 1 if dy > 0 else -1 if dy < 0 else 0

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = x_sign, 0, 0, y_sign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, y_sign, x_sign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        px = x0 + x * xx + y * yx
        py = y0 + x * xy + y * yy
        points.append((px, py))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return points

def idx2state(idx):
    """Converts a given index into the corresponding state.

    Args:
        idx (int): index of the entry whose state is required

    Returns:
        np.array: (x,y,x,y) state corresponding to the given index
    """
    state = np.empty(4)

    for i, j in enumerate(
        [
            Constants.M,
            Constants.N,
            Constants.M,
            Constants.N,
        ]
    ):
        state[i] = idx % j
        idx = idx // j
    return state


def state2idx(state):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = 0

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += state[i] * factor
        factor *= j

    return idx

def state2idx_vectorized(states):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = np.array([0]*states.shape[0])

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += states[:,i] * factor
        factor *= j

    return idx

def angle2movement(input_space,angle):
    if 5/8*np.pi <= angle < 7/8*np.pi:
        # North-West
        return input_space[0]
    elif 3/8*np.pi <= angle < 5/8*np.pi:
        # North
        return input_space[1]
    elif 1/8*np.pi <= angle < 3/8*np.pi:
        # North-East
        return input_space[2]
    elif angle >= 7/8*np.pi or angle < -7/8*np.pi:
        # West
        return input_space[3]
    elif -1/8*np.pi <= angle < 1/8*np.pi:
        # East
        return input_space[5]
    elif -7/8*np.pi <= angle < -5/8*np.pi:
        # South-West
        return input_space[6]
    elif -5/8*np.pi <= angle < -3/8*np.pi:
        # South
        return input_space[7]
    elif -3/8*np.pi <= angle < -1/8*np.pi:
        # South-East
        return input_space[8]
    else:
        return input_space[4]

def angle2idx(angle):
    if 5/8*np.pi <= angle < 7/8*np.pi:
        # North-West
        return 0
    elif 3/8*np.pi <= angle < 5/8*np.pi:
        # North
        return 1
    elif 1/8*np.pi <= angle < 3/8*np.pi:
        # North-East
        return 2
    elif angle >= 7/8*np.pi or angle < -7/8*np.pi:
        # West
        return 3
    elif -1/8*np.pi <= angle < 1/8*np.pi:
        # East
        return 5
    elif -7/8*np.pi <= angle < -5/8*np.pi:
        # South-West
        return 6
    elif -5/8*np.pi <= angle < -3/8*np.pi:
        # South
        return 7
    elif -3/8*np.pi <= angle < -1/8*np.pi:
        # South-East
        return 8
    else:
        return 4
