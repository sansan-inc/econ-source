import numpy as np
from scipy.special import softmax


def get_discounted_value(current_utility, discount_factor, transition_matrix, v):
    """

    :param current_utility: an array of shape (n_choices, n_states, 1) representing the result of evaluating the utility
    function at some parameter values.
    :param discount_factor: a float scalar in the range [0, 1) representing the discount factor of the agent.
    :param transition_matrix: an array of transition matrices with shape (n_choices, n_states, n_states).
    :param v: an array of shape (n_states, 1) representing the discounted expectation of the future value at each state.
    :return: a numpy array of shape (n_choices, n_states, 1)
    """
    n_choices, n_states, _ = transition_matrix.shape
    discounted_value = current_utility + discount_factor * np.array(
        [transition_matrix[i].dot(v) for i in range(n_choices)])
    return discounted_value


def phi_map(p, transition_matrix, parameters, utility_function, discount_factor, state_manager):
    """Mapping from the probability space to the value space. Assumes a Type I Extreme Value distribution for the
    unobservable component of the utility.
    :param p: the conditional choice probability numpy array with shape (n_choices, n_states, 1)
    :param transition_matrix: an array of transition matrices with shape (n_choices, n_states, n_states)
    :param parameters: the structural parameter values.
    :param utility_function: a function that takes as arguments an array of structural parameters, a set of choices and
    a mesh of state variables, and returns a numpy array of shape (n_choices, n_states, 1) that represents the utility
    value at each state and choice combination.
    :param discount_factor: a float scalar in the range [0, 1) representing the discount factor of the agent.
    :param state_manager: an instance of `StateManager`.
    :return:
    """
    n_choices, n_states, _ = p.shape
    current_utility = utility_function(
        parameters,
        np.arange(n_choices).reshape(-1, 1, 1),
        state_manager.get_states_mesh()
    )
    denominator = np.identity(n_states) - discount_factor*((p*transition_matrix).sum(axis=0))
    denominator = np.linalg.solve(denominator, np.identity(n_states))
    numerator = (p*(current_utility + np.euler_gamma - np.nan_to_num(np.log(p), 0))).sum(axis=0)
    v = denominator.dot(numerator)
    v = v - v.min()
    discounted_value = get_discounted_value(current_utility, discount_factor, transition_matrix, v)
    return discounted_value


def lambda_map(v):
    """Mapping from the value space to the probability space. Assumes a Type I Extreme Value distribution for the
    unobservable component of the utility.
    :param v:
    :return: a numpy array of shape (n_choices, n_states, 1) representing a conditional choice probability consistent
    with v and the distributional parametric assumption.
    """
    return softmax(v, axis=0)
