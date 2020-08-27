import numpy as np
from source_ddc.simulation_tools import simulate
from source_ddc.algorithms import CCP
from source_ddc.probability_tools import StateManager


def test_simulate_forward(simple_transition_matrix):
    def utility_fn(theta, choices, states):
        m_states, m_actions = np.meshgrid(states, choices)
        return (theta[0] * np.log(m_states + 1) - theta[1] * m_actions).reshape((len(choices), -1, 1))

    true_params = [0.5, 3]
    discount_factor = 0.95
    n_choices = 2
    n_states = 5
    n_simulation_draws = 10

    state_manager = StateManager(miles=n_states)

    df, ccp = simulate(100,
                       100,
                       n_choices,
                       state_manager,
                       true_params,
                       utility_fn,
                       discount_factor,
                       simple_transition_matrix)

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

    expected_shape = (n_simulation_draws + 1, df['action'].values.shape[0], 2)

    history = solver.simulate_forward(10)
    assert history.shape == expected_shape
