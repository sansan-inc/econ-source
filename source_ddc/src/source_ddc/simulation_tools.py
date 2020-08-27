import numpy as np
import pandas as pd
from .probability_tools import random_ccp
from .fixed_point import phi_map, lambda_map


def simulate(n_periods,
             n_agents,
             n_choices,
             state_manager,
             parameters,
             utility_function,
             discount_factor,
             transition_matrix,
             convergence_criterion=10e-6,
             seed=None):
    """A helper function that creates a simulated dataset for a Dynamic Discrete Choice model compatible with the
    parameters in the arguments.

    :param n_periods: an `int` representing the number of simulated periods for each agent.
    :param n_agents: an `int` representing the number of agents to be simulated.
    :param n_choices: an `int` representing the number of choices available to the agent.
    :param state_manager: an instance of `StateManager`.
    :param parameters: a list or numpy array containing the structural parameters
    :param utility_function: a function that takes as arguments an array of structural parameters, a set of choices and
    a mesh of state variables, and returns a numpy array of shape (n_choices, n_states, 1) that represents the utility
    value at each state and choice combination.
    :param discount_factor: a float scalar in the range [0, 1) representing the discount factor of the agent.
    :param transition_matrix: an array of transition matrices with shape (n_choices, n_states, n_states)
    :param parameters: the structural parameter values
    :param convergence_criterion: a tolerance level to determine the convergence of the iterations of the conditional
    choice probability array.
    :param seed: the seed for random number generation.
    :return: a tuple of pandas `DataFrame` and numpy array. The dataframe contains the simulated states and choices and
    the numpy array has shape (n_choices, n_states, 1) and represents the conditional choice probabilities.
    """

    if seed is not None:
        np.random.seed(seed)

    n_states = state_manager.total_states
    p = random_ccp(n_states, n_choices)
    converged = False

    #  Obtain the conditional choice probabilities by iterating until the fixed point is reach in probability space
    while not converged:
        p_0 = p
        v = phi_map(p, transition_matrix, parameters, utility_function, discount_factor, state_manager)
        p = lambda_map(v)
        delta = np.abs(np.max((p - p_0)))
        if delta <= convergence_criterion:
            converged = True

    errors = np.random.gumbel(size=(n_periods, n_agents, n_choices))
    agents, periods = [i.T.ravel() for i in np.meshgrid(np.arange(n_agents), np.arange(n_periods))]

    states = []
    actions = []

    for agent in range(n_agents):
        #     Draw some random initial state
        s = np.random.choice(np.arange(n_states))
        #         v = phi_map(p, transition_matrix, parameters, utility_function, discount_factor, state_manager)
        for t in range(n_periods):
            states.append(s)
            action = (errors[t, agent, :] + v[:, s, :].ravel()).argmax()
            actions.append(action)
            if t != n_periods:
                s = np.random.choice(list(range(n_states)), p=transition_matrix[action, s])

    df = pd.DataFrame({
        'agent_id': agents,
        't': periods,
        'state': states,
        'action': actions
    })

    return df, p


def simulate_state_draw(current_action_state, transition_matrix):
    """Convenience function for obtaining the following state given a departure point n the state space and a
    transition matrix.

    :param current_action_state: an array of current statess.
    :param transition_matrix: a numpy array of shape (n_choices, n_states_ 1) representing transition probabilities/
    :return: an array of future states.
    """
    next_states = np.empty(current_action_state.shape[1]).astype(np.int32)
    for i in range(current_action_state[0].shape[0]):
        s = current_action_state[0][i]
        transition_probs = transition_matrix[s[0], s[1]]
        next_states[i] = np.searchsorted(np.cumsum(transition_probs), np.random.random(), side="right")
    return next_states


def simulate_action_draw(ccp, states):
    """Convenience function for simulating an action draw given a decision policy in the form of conditional choice
    probabilities.

    :param ccp: the conditional choice probabilities as a numpy array of shape (n_choices, n_states, 1).
    :param states: an array of current states.
    :return:
    """
    return np.array([
        np.searchsorted(np.cumsum(ccp.reshape(ccp.shape[0], -1).T[s]), np.random.random(), side="right") for s in states
    ])
