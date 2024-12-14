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

    # List all states
    start_idx = np.arange(Constants.K)
    drone_states = idx2state_vectorized(start_idx)
    # Computed state idxs for all unreachable states
    # Crash drone with stationary drone
    crash_drone_idx = np.any(np.all(Constants.DRONE_POS[:,None] == drone_states[:,0:2], axis=2), axis=0)
    # Terminal state
    goal_idx = np.all(drone_states[:,0:2] == Constants.GOAL_POS, axis=1)
    # Crash drone with swan
    crash_swan_idx = np.all(drone_states[:,:2] == drone_states[:,2:], axis=1)
    # Filter out unreachable states
    start_idx = start_idx[~crash_drone_idx & ~goal_idx & ~crash_swan_idx]
    # Set Q for unreachable states to zero
    Q[crash_drone_idx] = 0
    Q[goal_idx] = 0
    Q[crash_swan_idx] = 0
    # For remaining states, set Q to the cost of moving to the goal
    Q[start_idx] = Constants.TIME_COST + Constants.THRUSTER_COST * np.sum(np.abs(Constants.INPUT_SPACE), axis=1)

    for curr_state_idx in start_idx:
        curr_state = drone_states[curr_state_idx].astype(int)
        curr_state_drone = curr_state[:2]
        curr_state_swan = curr_state[2:]

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
