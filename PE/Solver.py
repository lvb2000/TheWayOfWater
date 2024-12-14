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
from scipy import optimize

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
    np.random.seed(42)
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
    J_opt = init_values_with_distance(Constants)

    K, _, L = P.shape  # Number of states (K) and control inputs (L)

    while True:
        J_old = J_opt.copy()
        # This uses the Gauss-Seidel update rule
        for i in range(Constants.K):
            costs = Q[i, :] + np.einsum('jk,j->k',P[i,:,:],J_opt)
            J_opt[i] = min(costs)  # Optimal value for state i

        if np.allclose(J_old, J_opt,rtol=1e-04,atol=1e-07):
            break

    # Derive optimal policy
    costs = Q + np.einsum('ijk,j->ik', P, J_opt)
    u_opt = np.argmin(costs,axis=1)  # Choose the action minimizing the cost

    return J_opt, u_opt

def policy_iteration(P, Q, Constants):
    J_opt = init_values_with_distance(Constants)
    u_opt = init_towards_goal(Constants)
    idx = np.arange(Constants.K)
    eye_k = np.eye(Constants.K)

    while True:
        # Policy Evaluation
        A = eye_k - P[idx, :, u_opt]  # Transition matrix for current policy
        b = Q[idx, u_opt]            # Rewards for current policy
        J_new = np.linalg.solve(A, b)

        # Convergence Check
        if np.allclose(J_new, J_opt,rtol=1e-04,atol=1e-07):
            break
        J_opt = J_new

        # Policy Improvement
        cost_matrix = Q + np.einsum('ijk,j->ik', P, J_opt)  # Compute costs
        u_opt = np.argmin(cost_matrix, axis=1)

    return J_opt, u_opt

def linear_programming(P, Q, Constants):
    c = np.ones(Constants.K).T
    # get all possible actions
    eye_k = np.eye(Constants.K)
    A = (eye_k[:, :, None] - P).transpose(2, 0, 1).reshape(-1, Constants.K)
    b = Q.transpose(1, 0).ravel()
    l = np.zeros(Constants.K)
    u = np.ones(Constants.K) * np.inf
    res = optimize.linprog(c=-c, A_ub=A, b_ub=b, bounds=list(zip(l, u)))
    costs = Q + np.einsum('ijk,j->ik', P, res.x)
    u_opt = np.argmin(costs, axis=1)  # Choose the action minimizing the cost
    return res.x, u_opt

def hybrid(P, Q, Constants):
    J_opt = np.zeros(Constants.K)

    # init random policies for exploration during first step
    u_opt = init_towards_goal(Constants)
    idx = np.arange(Constants.K)

    while True:
        # value evaluation
        J_new = Q[idx, u_opt] + np.einsum('ij,j->i', P[idx, :, u_opt], J_opt)
        J_new = Q[idx, u_opt] + np.einsum('ij,j->i', P[idx, :, u_opt], J_new)
        J_new = Q[idx, u_opt] + np.einsum('ij,j->i', P[idx, :, u_opt], J_new)
        # Convergence Check
        if np.allclose(J_new, J_opt,rtol=1e-04,atol=1e-07):
            break
        J_opt = J_new
        # policy improvement
        u_opt = np.argmin(Q + np.einsum('ijk,j->ik', P, J_opt), axis=1)

    return J_opt, u_opt

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

def init_values_with_distance(Constants):
    goal = Constants.GOAL_POS
    idx = np.arange(Constants.K)
    pos = idx2state_vectorized(idx)
    # chebyshev distance which treats the diagonal as 1
    distance = np.max(np.abs(pos[:,:2] - goal), axis=1)
    return distance