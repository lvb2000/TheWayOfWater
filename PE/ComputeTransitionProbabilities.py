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
        # loop over all possible inputs
        for input_idx in range(Constants.L):
            # Handle Drone movement
            # handle stay input
            input = Constants.INPUT_SPACE[input_idx]
            # next state without disturbance
            next_state_drone = curr_state_drone + input
            # with disturbance currents
            prob_current = Constants.CURRENT_PROB[tuple(curr_state_drone)]
            applied_current = Constants.FLOW_FIELD[tuple(curr_state_drone)]
            # next drone state probabilities
            possible_next_states_drone = [next_state_drone, next_state_drone + applied_current]
            possible_next_states_prob_drone = [1 - prob_current, prob_current]

            # Check if drone is out of bounds before handling swan movement
            possible_next_states_drone_inbound = []
            possible_next_states_prob_drone_inbound = []
            for idx in range(len(possible_next_states_drone)):
                P, inbound = check_bounds(idx, P, Constants, reset_states, reset_prob, possible_next_states_drone, possible_next_states_prob_drone, curr_state_idx, input_idx)
                if inbound:
                    P, crash = check_crash(idx, P, Constants, reset_states, reset_prob, curr_state_drone, possible_next_states_drone,
                                           possible_next_states_prob_drone, curr_state_idx, input_idx)
                if inbound and not crash:
                    possible_next_states_drone_inbound.append(possible_next_states_drone[idx])
                    possible_next_states_prob_drone_inbound.append(possible_next_states_prob_drone[idx])
            # If all options are out of bounds, skip swan analysis
            if len(possible_next_states_drone_inbound) == 0:
                continue

            # Handle Swan movement
            # determine the angle to drone
            delta_y = curr_state_swan[1] - curr_state_drone[1]
            delta_x = curr_state_drone[0] - curr_state_swan[0]
            angle_to_drone = np.arctan2(delta_y, delta_x)
            movement_to_drone = angle2movement(Constants.INPUT_SPACE, angle_to_drone)

            possible_next_states_swan = [curr_state_swan, curr_state_swan + movement_to_drone]
            possible_next_states_prob_swan = [1 - Constants.SWAN_PROB, Constants.SWAN_PROB]

            # combine drone and swan states and add to transition matrix
            for next_state_drone, prob_drone in zip(possible_next_states_drone_inbound, possible_next_states_prob_drone_inbound):
                for next_state_swan, prob_swan in zip(possible_next_states_swan, possible_next_states_prob_swan):
                    # check if drone and swan are at the same position in next state if yes reset
                    if np.all(next_state_drone == next_state_swan):
                        P = apply_reset(P, reset_states, prob_drone * prob_swan * reset_prob, curr_state_idx, input_idx)
                        continue
                    next_state = np.concatenate((next_state_drone, next_state_swan))
                    next_state_idx = state2idx(next_state)
                    P[curr_state_idx, next_state_idx, input_idx] += prob_drone * prob_swan

    return P

def check_crash(idx,P, Constants, reset_states, reset_prob, curr_state_drone, possible_next_states_drone, possible_next_states_prob, curr_state_idx, input_idx):
    # check if crash with static drones by bresenham function
    # get path
    path = bresenham(curr_state_drone, possible_next_states_drone[idx])
    for pos in path:
        if np.any(np.all(Constants.DRONE_POS == pos, axis=1)):
            # apply reset to transition probability matrix
            P = apply_reset(P, reset_states, possible_next_states_prob[idx] * reset_prob, curr_state_idx, input_idx)
            return P, True
    return P, False
def check_bounds(idx,P, Constants, reset_states, reset_prob, possible_next_states_drone, possible_next_states_prob, curr_state_idx, input_idx):
    if (np.any(possible_next_states_drone[idx] < 0)
            or possible_next_states_drone[idx][0] >= Constants.M
            or possible_next_states_drone[idx][1] >= Constants.N):
        # apply reset to transition probability matrix
        P = apply_reset(P, reset_states, possible_next_states_prob[idx] * reset_prob, curr_state_idx, input_idx)
        # remove this option
        return P, False
    return P, True
def apply_reset(P, reset_states, reset_prob, curr_state_idx, input_idx):
    P[curr_state_idx, reset_states, input_idx] += reset_prob
    return P

def get_reset_state(Constants):
    idx = []
    # get all index where drone is at starting position and swan everywhere else
    for i in range(Constants.M):
        for j in range(Constants.N):
            if i!=Constants.START_POS[0] or j!=Constants.START_POS[1]:
                idx.append(state2idx(np.array([Constants.START_POS[0], Constants.START_POS[1], i, j])))
    return np.array(idx),1/len(idx)

