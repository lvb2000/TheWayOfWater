"""
 ComputeExpectedStageCosts.py

 Python function template to compute the expected stage cost.

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
from utils import *


def compute_expected_stage_cost(Constants):
    """Computes the expected stage cost for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L)
    """
    Q = np.ones((Constants.K, Constants.L)) * np.inf
    input_idx = np.array([np.arange(Constants.L),np.arange(Constants.L)]).flatten()

    # TODO fill the expected stage cost Q here
    for curr_state_idx in range(Constants.K):
        curr_state = idx2state(curr_state_idx).astype(int)
        curr_state_drone = curr_state[:2]
        curr_state_swan = curr_state[2:]
        # it is not possible for the drone to be already crashed
        if np.any(np.all(Constants.DRONE_POS == curr_state_drone, axis=1)):
            Q[curr_state_idx, :] = 0
            continue
        # check if drone reached goal
        if np.all(idx2state(curr_state_idx)[:2] == Constants.GOAL_POS):
            # Terminal cost
            Q[curr_state_idx, :] = 0
            continue
        # Swan and drone being at the same position will never happen as the game is reset before
        if np.all(curr_state_drone == curr_state_swan):
            Q[curr_state_idx, :] = 0
            continue

        # add thruster cost
        Q[curr_state_idx, :] = Constants.TIME_COST + Constants.THRUSTER_COST * np.sum(np.abs(Constants.INPUT_SPACE),axis=1)

        # next state without disturbance
        next_state_drone = curr_state_drone + Constants.INPUT_SPACE
        # with disturbance currents
        prob_current = Constants.CURRENT_PROB[tuple(curr_state_drone)]
        applied_current = Constants.FLOW_FIELD[tuple(curr_state_drone)]
        # next drone state probabilities
        possible_next_states_drone = np.array([next_state_drone, next_state_drone + applied_current]).reshape(-1,2)
        possible_next_states_drone_prob = np.array([[1 - prob_current] * Constants.L, [prob_current] * Constants.L]).flatten()

        # Boundary and crash checks (vectorized)
        inbound_mask = check_bounds_vectorized(possible_next_states_drone, Constants)
        # Check crashes only fore inbound states
        crash_mask = np.array([False] * possible_next_states_drone.shape[0])
        if np.any(inbound_mask):
            crash_mask[inbound_mask] = check_crash_vectorized(possible_next_states_drone[inbound_mask], curr_state_drone, Constants)

        # Filter inbound and non-crash states
        valid_mask = inbound_mask & ~crash_mask
        # Update Q for invalid states
        Q[curr_state_idx, :] += Constants.DRONE_COST * ((1 - prob_current) * (~valid_mask)[0:9] + prob_current * (~valid_mask)[9:])
        if not np.any(valid_mask):
            continue

        input_mask = input_idx[valid_mask]
        valid_next_states_drone = possible_next_states_drone[valid_mask]
        valid_next_states_drone_prob = possible_next_states_drone_prob[valid_mask]

        # Handle Swan movement
        # determine the angle to drone
        delta_y = curr_state_swan[1] - curr_state_drone[1]
        delta_x = curr_state_drone[0] - curr_state_swan[0]
        angle_to_drone = np.arctan2(delta_y, delta_x)
        movement_to_drone = angle2movement(Constants.INPUT_SPACE, angle_to_drone)

        drone_swan_crash = np.all(valid_next_states_drone[:, None] == np.array([curr_state_swan, curr_state_swan + movement_to_drone]), axis=2)
        prob = (1 - Constants.SWAN_PROB) * valid_next_states_drone_prob * drone_swan_crash[:,0] + Constants.SWAN_PROB * valid_next_states_drone_prob * drone_swan_crash[:,1]
        np.add.at(Q[curr_state_idx], input_mask, prob * Constants.DRONE_COST)

    return Q


def check_crash(idx, Q, Constants, curr_state_drone, possible_next_states_drone,
                possible_next_states_prob, curr_state_idx, input_idx):
    # check if crash with static drones by bresenham function
    # get path
    path = bresenham(curr_state_drone, possible_next_states_drone[idx])
    matching_indices = np.any(np.all(Constants.DRONE_POS[:, None] == path, axis=2), axis=0)

    if np.any(matching_indices):
        # Apply reset
        Q[curr_state_idx, input_idx] += Constants.DRONE_COST * possible_next_states_prob[idx]
        return Q, True
    return Q, False


def check_bounds(idx, Q, Constants, possible_next_states_drone, possible_next_states_prob,
                 curr_state_idx, input_idx):
    if (np.any(possible_next_states_drone[idx] < 0)
            or possible_next_states_drone[idx][0] >= Constants.M
            or possible_next_states_drone[idx][1] >= Constants.N):
        # apply reset to transition probability matrix
        Q[curr_state_idx,input_idx] += Constants.DRONE_COST * possible_next_states_prob[idx]
        # remove this option
        return Q, False
    return Q, True

def check_bounds_vectorized(possible_next_states_drone, Constants):
    return np.all(possible_next_states_drone >= 0, axis=1) & np.all(possible_next_states_drone < [Constants.M, Constants.N], axis=1)

def check_crash_vectorized(possible_next_states_drone, curr_state_drone, Constants):
    # check if crash with static drones by bresenham function
    ret = np.zeros(possible_next_states_drone.shape[0], dtype=bool)
    for idx,next_state_drone in enumerate(possible_next_states_drone):
        path = bresenham(curr_state_drone, next_state_drone)
        matching_indices = np.any(np.all(Constants.DRONE_POS[:, None] == path, axis=2), axis=0)
        if np.any(matching_indices):
            ret[idx] = True
    return ret

def angle2movement(input_space, angle):
    if 5 / 8 * np.pi <= angle < 7 / 8 * np.pi:
        # North-West
        return input_space[0]
    elif 3 / 8 * np.pi <= angle < 5 / 8 * np.pi:
        # North
        return input_space[1]
    elif 1 / 8 * np.pi <= angle < 3 / 8 * np.pi:
        # North-East
        return input_space[2]
    elif angle >= 7 / 8 * np.pi or angle < -7 / 8 * np.pi:
        # West
        return input_space[3]
    elif -1 / 8 * np.pi <= angle < 1 / 8 * np.pi:
        # East
        return input_space[5]
    elif -7 / 8 * np.pi <= angle < -5 / 8 * np.pi:
        # South-West
        return input_space[6]
    elif -5 / 8 * np.pi <= angle < -3 / 8 * np.pi:
        # South
        return input_space[7]
    elif -3 / 8 * np.pi <= angle < -1 / 8 * np.pi:
        # South-East
        return input_space[8]
    else:
        return input_space[4]
