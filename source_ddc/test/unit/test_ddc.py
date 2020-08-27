import numpy as np
from source_ddc.simulation_tools import simulate
from source_ddc.algorithms import NFXP, NPL, CCP
from source_ddc.probability_tools import StateManager, random_ccp
from test.utils.functional_tools import average_out

n_repetitions = 10


def test_nfxp(simple_transition_matrix):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    @average_out(n_repetitions)
    def test():
        df, _ = simulate(
            500,
            100,
            n_choices,
            state_manager,
            true_params,
            utility_fn,
            discount_factor,
            simple_transition_matrix
        )

        algorithm = NFXP(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            parameter_names=['variable_cost', 'replacement_cost']
        )

        return algorithm.estimate(start_params=[-1, -1], method='bfgs')

    mean_params = test()
    tolerance_levels = np.array([0.05, 0.05])
    assert np.all(np.abs(mean_params - true_params) < tolerance_levels)


def test_ccp(simple_transition_matrix):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    @average_out(n_repetitions)
    def test():
        df, ccp = simulate(
            500,
            100,
            n_choices,
            state_manager,
            true_params,
            utility_fn,
            discount_factor,
            simple_transition_matrix
        )

        algorithm = CCP(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            parameter_names=['variable_cost', 'replacement_cost']
        )

        return algorithm.estimate(start_params=[1, 1], method='bfgs')

    mean_params = test()
    tolerance_levels = np.array([0.05, 0.05])
    assert np.all(np.abs(mean_params - true_params) < tolerance_levels)


def test_npl(simple_transition_matrix):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    @average_out(n_repetitions)
    def test():
        df, _ = simulate(
            500,
            100,
            n_choices,
            state_manager,
            true_params,
            utility_fn,
            discount_factor,
            simple_transition_matrix)

        ccp = random_ccp(n_states, n_choices)

        algorithm = NPL(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            parameter_names=['variable_cost', 'replacement_cost']
        )
        return algorithm.estimate(start_params=[1, 1], method='bfgs')

    mean_params = test()
    tolerance_levels = np.array([0.05, 0.05])
    assert np.all(np.abs(mean_params - true_params) < tolerance_levels)


def test_npl_relaxation_param(simple_transition_matrix):

    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5

    state_manager = StateManager(miles=n_states)

    @average_out(n_repetitions)
    def test():
        df, _ = simulate(500,
                         100,
                         n_choices,
                         state_manager,
                         true_params,
                         utility_fn,
                         discount_factor,
                         simple_transition_matrix)

        ccp = random_ccp(n_states, n_choices)

        algorithm = NPL(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            relaxation_param=0.9,
            parameter_names=['variable_cost', 'replacement_cost']
        )

        return algorithm.estimate(start_params=[1, 1], method='bfgs')

    mean_params = test()
    tolerance_levels = np.array([0.05, 0.05])
    assert np.all(np.abs(mean_params - true_params) < tolerance_levels)
