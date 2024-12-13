"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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

def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    P = np.zeros((Constants.K, Constants.K, Constants.L))

    # TODO fill the transition probability matrix P here
    reset_states, reset_prob = get_reset_state(Constants)
    input_idx = np.array([np.arange(Constants.L), np.arange(Constants.L)]).flatten()
    # loop over all current states
    for curr_state_idx in range(Constants.K):
        curr_state = idx2state(curr_state_idx).astype(int)
        curr_state_drone = curr_state[:2]
        curr_state_swan = curr_state[2:]
        # it is not possible for the drone to be already crashed
        if np.any(np.all(Constants.DRONE_POS == curr_state_drone, axis=1)):
            continue
        # If we are in the goal state, we stay there anyway
        if np.all(curr_state_drone == Constants.GOAL_POS):
            continue
        # Swan and drone being at the same position will never happen as the game is reset before
        if np.all(curr_state_drone == curr_state_swan):
            continue

        # next state without disturbance
        next_state_drone = curr_state_drone + Constants.INPUT_SPACE
        # with disturbance currents
        prob_current = Constants.CURRENT_PROB[tuple(curr_state_drone)]
        applied_current = Constants.FLOW_FIELD[tuple(curr_state_drone)]
        # next drone state probabilities
        possible_next_states_drone = np.array([next_state_drone, next_state_drone + applied_current]).reshape(-1, 2)
        possible_next_states_drone_prob = np.array([[1 - prob_current] * Constants.L, [prob_current] * Constants.L]).flatten()

        # Boundary and crash checks (vectorized)
        inbound_mask = check_bounds_vectorized(possible_next_states_drone, Constants)
        # Check crashes only fore inbound states
        crash_mask = np.array([False] * possible_next_states_drone.shape[0])
        if np.any(inbound_mask):
            crash_mask[inbound_mask] = check_crash_vectorized(possible_next_states_drone[inbound_mask],curr_state_drone, Constants)

        # Filter inbound and non-crash states
        valid_mask = inbound_mask & ~crash_mask
        # Update P for invalid states
        P[curr_state_idx,reset_states, :] += reset_prob * ((1 - prob_current) * (~valid_mask)[0:9] + prob_current * (~valid_mask)[9:])
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
        valid_next_states_swan = np.array([curr_state_swan, curr_state_swan + movement_to_drone])

        drone_swan_crash = np.all(valid_next_states_drone[:, None] == valid_next_states_swan, axis=2)
        prob_crash = ((1 - Constants.SWAN_PROB) * valid_next_states_drone_prob * drone_swan_crash[:,0]
                + Constants.SWAN_PROB * valid_next_states_drone_prob * drone_swan_crash[:,1])

        # apply reset to P
        row_indeces = np.repeat(reset_states, input_mask.shape[0])
        col_indeces = np.tile(input_mask, reset_states.shape[0])
        np.add.at(P[curr_state_idx], (row_indeces,col_indeces), np.array([prob_crash * reset_prob]*reset_states.shape[0]).flatten())

        next_drone_state_idx = state2idx_vectorized(np.pad(valid_next_states_drone, ((0, 0), (0, 2)), mode='constant'))
        next_swan_state_idx = state2idx_vectorized(np.pad(valid_next_states_swan, ((0, 0), (2, 0)), mode='constant'))

        next_state_idx = np.array(np.meshgrid(next_drone_state_idx, next_swan_state_idx)).T.reshape(-1,2,2).sum(axis=2)

        prob_no_crash = np.column_stack([
            (1 - Constants.SWAN_PROB) * valid_next_states_drone_prob * ~drone_swan_crash[:, 0],
            Constants.SWAN_PROB * valid_next_states_drone_prob * ~drone_swan_crash[:, 1]
        ])

        np.add.at(P[curr_state_idx], (next_state_idx.flatten(), np.repeat(input_mask,2)), prob_no_crash.flatten())
    return P

def check_crash_vectorized(possible_next_states_drone, curr_state_drone, Constants):
    # check if crash with static drones by bresenham function
    ret = np.zeros(possible_next_states_drone.shape[0], dtype=bool)
    for idx,next_state_drone in enumerate(possible_next_states_drone):
        path = bresenham(curr_state_drone, next_state_drone)
        matching_indices = np.any(np.all(Constants.DRONE_POS[:, None] == path, axis=2), axis=0)
        if np.any(matching_indices):
            ret[idx] = True
    return ret
def check_bounds_vectorized(possible_next_states_drone, Constants):
    return np.all(possible_next_states_drone >= 0, axis=1) & np.all(possible_next_states_drone < [Constants.M, Constants.N], axis=1)

def get_reset_state(Constants):
    idx = []
    # get all index where drone is at starting position and swan everywhere else
    for i in range(Constants.M):
        for j in range(Constants.N):
            if i!=Constants.START_POS[0] or j!=Constants.START_POS[1]:
                idx.append(state2idx(np.array([Constants.START_POS[0], Constants.START_POS[1], i, j])))
    return np.array(idx),1/len(idx)

