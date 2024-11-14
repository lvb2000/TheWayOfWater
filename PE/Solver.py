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

    J_opt = np.zeros(Constants.K)
    u_opt = np.zeros(Constants.K, dtype=int)

    K, _, L = P.shape  # Number of states (K) and control inputs (L)
    gamma = 1.0
    tol = 1e-6

    count = 0

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

        if count > 100:
            print(J_new)
            count = 0
        count += 1
        J_opt = J_new

    # Derive optimal policy
    for i in range(Constants.K):
        costs = []
        for l in range(L):
            cost = Q[i, l] + gamma * np.sum(P[i, :, l] * J_opt)
            costs.append(cost)
        u_opt[i] = np.argmin(costs)  # Choose the action minimizing the cost

    return J_opt, u_opt
