# TheWayOfWater

### Water-Based Drone Navigation for Light Show in Lake Zürich

This project involves safely navigating a main drone to its designated position as part of a drone light show in Lake
Zürich. The main challenges include avoiding collisions with stationary drones and evading interference from a curious
swan. Stationary drones are equipped with GPS and remain unaffected by collisions, but the main drone sinks if a
collision occurs, requiring deployment of a replacement.\
Note: All details are explained in the PE.pdf file.

### Key Tasks

1. **Transition Probabilities**: Implement the function `compute_transition_probabilities`
   in `ComputeTransitionProbabilities.py` to calculate the transition probability matrix $\( P \in \mathbb{R}^{K \times
   K \times L} \)$, where:
    - $\( K \)$: Number of possible states.
    - $\( L \)$: Number of possible inputs.
    - $\( P[i, j, k] \)$: Probability of transitioning from state $\( stateSpace[i] \)$ to $\(
      stateSpace[j] \)$ using input $\( inputSpace[k] \)$.

2. **Expected Stage Costs**: Implement the function `compute_expected_stage_costs` in `ComputeExpectedStageCosts.py` to
   compute the expected stage cost matrix $\( Q \in \mathbb{R}^{K \times L} \)$, where:
    - $\( Q[i, k] \)$: Expected cost of applying input $\( inputSpace[k] \)$ at state $\(
      stateSpace[i] \)$.

3. **Optimal Policy and Costs**: Implement the function `solution` in `Solver.py` to calculate:
    - The optimal cost $\( J \in \mathbb{R}^K \)$, where $\( J[i] \)$ is the cost incurred starting from state $\( i \)$
      with the optimal policy.
    - The optimal policy $\( \text{policy} \in \mathbb{N}^K \)$, where $\( \text{policy[i]} \)$ specifies the optimal
      input for $\( stateSpace[i] \)$.

### RunTime Analysis and Ranking

As this project is set up to be ranked competitively, the runtime of the code is crucial, as it is the primary factor in
the ranking. All three of the above-mentioned tasks are implemented with respect to the runtime. To achieve this, I
tried using as much of numpys vectorized operations as possible. Especially the np.einsum function gave me a significant
speedup in the computation of the solver function instead of using np.sum(np.multiply(...)). As einsum uses the einstein
summation convention and stores the intermediate results it is more efficient in memory usage and thereby faster in
computation. \
A function by function analysis of the runtime can be found in the baseline.pdf and handin4.pdf file, which show the
runtime of the first run of the code and the final runtime of the code after optimization. \
For the solver I tested four different approaches to solve the problem. I tried value iteration, policy iteration, a
hybrid of the two and linear programming with the scipy optimization library. The policy iteration was the fastest in
end.



