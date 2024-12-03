"""
 Solver.py

 Python function template to solve the stochastic
 shortest path problem.

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

ALGORITHM = "Policy Iteration"

def solution(P, Q, Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        Q  (np.array): A (K x L)-matrix containing the expected stage costs of all states
                       in the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP
        np.array: The optimal control policy for the stochastic SPP

    """
    if ALGORITHM == "Value Iteration":
        return value_iteration(P, Q, Constants)
    elif ALGORITHM == "Policy Iteration":
        return policy_iteration(P, Q, Constants)
    elif ALGORITHM == "Linear Programming":
        return linear_programming(P, Q, Constants)
    elif ALGORITHM == "Hybrid":
        return hybrid(P, Q, Constants)
    return None, None


def value_iteration(P, Q, Constants):
    J_opt = np.zeros(Constants.K)
    u_opt = np.zeros(Constants.K, dtype=int)

    K, _, L = P.shape  # Number of states (K) and control inputs (L)
    gamma = 1.0
    tol = 1e-6

    while True:
        J_new = np.zeros(Constants.K)
        for i in range(Constants.K):
            costs = []
            for l in range(L):
                # Compute cost for taking control l in state i
                # check if cost is infinte
                cost = Q[i, l] + gamma * np.sum(P[i, :, l] * J_opt)
                costs.append(cost)
            J_new[i] = min(costs)  # Optimal value for state i

        if np.max(np.abs(J_new - J_opt)) < tol:
            break

        J_opt = J_new

    # Derive optimal policy
    for i in range(Constants.K):
        costs = []
        for l in range(L):
            cost = Q[i, l] + gamma * np.sum(P[i, :, l] * J_opt)
            costs.append(cost)
        u_opt[i] = np.argmin(costs)  # Choose the action minimizing the cost

    return J_opt, u_opt

def policy_iteration(P, Q, Constants):
    tol = 1e-6

    J_opt = np.zeros(Constants.K)

    # Initialize policy with direction towards goal
    u_opt = init_towards_goal(Constants)
    idx = np.arange(u_opt.shape[0])

    while True:
        J_new = np.zeros(Constants.K)
        # policy evaluation
        J_new = policy_evaluation(Q, P, J_opt, u_opt, idx)
        J_new = policy_evaluation(Q, P, J_new, u_opt, idx)
        J_new = policy_evaluation(Q, P, J_new, u_opt, idx)
        # policy improvement (u_opt gets updated)
        u_opt = policy_improvement(Q,P,J_new)
        # converging condition
        if np.max(np.abs(J_new-J_opt)) < tol:
            break
        # update
        J_opt = J_new

    return J_opt, u_opt
def policy_evaluation(Q,P,J_opt,u_opt, idx):
    return Q[idx, u_opt] + np.sum(P[idx, :, u_opt] * J_opt, axis=1)
def policy_improvement(Q,P,J_new):
    return np.argmin(Q + np.einsum('ijk,k->ij', P.transpose(0, 2, 1), J_new),axis=1)

def init_towards_goal(Constants):
    goal = Constants.GOAL_POS
    u_opt = np.zeros(Constants.K, dtype=int)
    for i in range(Constants.K):
        pos = idx2state(i)
        delta_y = pos[1] - goal[1]
        delta_x = goal[0] - pos[0]
        angle_to_drone = np.arctan2(delta_y, delta_x)
        u_opt[i] = angle2idx(angle_to_drone)
    return u_opt

def linear_programming(P, Q, Constants):
    return None, None

def hybrid(P, Q, Constants):
    return None, None