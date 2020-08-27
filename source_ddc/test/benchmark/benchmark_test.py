import numpy as np
from source_ddc.simulation_tools import simulate
from source_ddc.algorithms import NFXP, CCP, NPL
from source_ddc.probability_tools import StateManager, random_ccp

n_agents = 1000
n_periods = 100


def test_profile_nfxp(simple_transition_matrix, benchmark):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    df, _ = simulate(n_periods,
                     n_agents,
                     n_choices,
                     state_manager,
                     true_params,
                     utility_fn,
                     discount_factor,
                     simple_transition_matrix)

    def estimate():
        solver = NFXP(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            parameter_names=['variable_cost', 'replacement_cost']
        )
        solver.estimate(start_params=[1, 1], method='bfgs')

    benchmark(estimate)


def test_profile_ccp(simple_transition_matrix, benchmark):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    df, ccp = simulate(n_periods,
                       n_agents,
                       n_choices,
                       state_manager,
                       true_params,
                       utility_fn,
                       discount_factor,
                       simple_transition_matrix)

    def estimate():
        solver = CCP(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            parameter_names=['variable_cost', 'replacement_cost']
        )
        solver.estimate(start_params=[1, 1], method='bfgs')

    benchmark(estimate)


def test_profile_npl(simple_transition_matrix, benchmark):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    df, _ = simulate(n_periods,
                     n_agents,
                     n_choices,
                     state_manager,
                     true_params,
                     utility_fn,
                     discount_factor,
                     simple_transition_matrix)

    ccp = random_ccp(n_states, n_choices)

    def estimate():
        solver = NPL(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            parameter_names=['variable_cost', 'replacement_cost']
        )
        solver.estimate(start_params=[1, 1], method='bfgs')

    benchmark(estimate)
