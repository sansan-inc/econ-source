import numpy as np
from source_ddc.simulation_tools import simulate
from source_ddc.algorithms import NPL, CCP
from source_ddc.probability_tools import StateManager, random_ccp
from test.utils.functional_tools import average_out

n_repetitions = 10


def test_state_manager(simple_transition_matrix):

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
        df, ccp = simulate(100,
                           100,
                           n_choices,
                           state_manager,
                           true_params,
                           utility_fn,
                           discount_factor,
                           simple_transition_matrix)

        solver = NPL(
            df['action'].values,
            df['state'].values,
            simple_transition_matrix,
            utility_fn,
            discount_factor,
            initial_p=ccp,
            parameter_names=['variable_cost', 'replacement_cost'],
            state_manager=state_manager)
        return solver.estimate(start_params=[1, 1], method='bfgs')

    mean_params = test()
    tolerance_levels = np.array([0.1, 0.1])
    assert np.all(np.abs(mean_params - true_params) < tolerance_levels)
    assert state_manager.get_states_mesh().shape == (1, n_states, 1)


def test_multivariate_utility_fn_estimation(large_transition_matrix):

    unit_value_real = 4
    exit_unit_cost = -3
    exit_unit_value = unit_value_real + exit_unit_cost
    plan_a_unit_cost = -2.5
    plan_a_fixed_cost = -4
    plan_b_unit_cost = -2
    plan_b_unit_value = unit_value_real + plan_b_unit_cost
    plan_b_user_cost = -5

    real_params = [
        unit_value_real,
        exit_unit_cost,
        plan_a_unit_cost,
        plan_a_fixed_cost,
        plan_b_unit_cost,
        plan_b_user_cost
    ]
    discount_factor = 0.9
    n_choices = 3
    n_periods = 100
    n_agents = 1000

    def real_utility_fn(theta, _, states):
        [
            unit_value,
            unit_exit_cost,
            unit_plan_a_cost,
            fixed_plan_a_cost,
            unit_plan_b_cost,
            user_plan_b_cost
        ] = theta

        [cards, users] = states
        u_exit = (unit_value + unit_exit_cost) * cards
        u_plan_a = (unit_value + unit_plan_a_cost) * cards + fixed_plan_a_cost
        u_plan_b = (unit_value + unit_plan_b_cost) * cards + user_plan_b_cost * users
        u = np.array([u_exit, u_plan_a, u_plan_b])
        return u

    def model_utility_fn(theta, _, states):
        [exit_net_unit_value, unit_value, plan_b_net_unit_value] = theta
        [units, users] = states
        u_exit = exit_net_unit_value * units
        u_plan_a = (unit_value + plan_a_unit_cost) * units + plan_a_fixed_cost
        u_plan_b = plan_b_net_unit_value * units + plan_b_user_cost * users
        u = np.array([u_exit, u_plan_a, u_plan_b])
        return u

    state_manager = StateManager(units=5, users=2)

    @average_out(n_repetitions)
    def test():
        df, ccp = simulate(
            n_periods,
            n_agents,
            n_choices,
            state_manager,
            real_params,
            real_utility_fn,
            discount_factor,
            large_transition_matrix,
            convergence_criterion=10e-6)

        parameter_names = ['exit_net_unit_value', 'unit_value', 'plan_b_net_unit_value']

        algo = NPL(df['action'].values,
                   df['state'].values,
                   large_transition_matrix,
                   model_utility_fn,
                   discount_factor,
                   initial_p=random_ccp(state_manager.total_states, 3),
                   state_manager=state_manager,
                   parameter_names=parameter_names
                   )
        return algo.estimate(start_params=np.random.uniform(size=len(parameter_names)), method='bfgs')

    mean_params = test()

    tolerance_levels = np.array([0.1, 0.1, 0.1])
    assert np.all(np.abs(mean_params - [exit_unit_value, unit_value_real, plan_b_unit_value]) < tolerance_levels)


def test_multivariate_ddc():
    n_age_states = 30
    n_health_states = 2
    n_retirement_states = 2
    n_pension_states = 3
    n_medical_exp_states = 3
    discount_factor = 0.9

    def utility_fn(theta, _, states):
        [age, health, retirement, pension, medical_exp] = states
        u_retires = pension - medical_exp

        wage = np.exp(theta[0] + theta[1] * health + theta[2] * (age / (1 + age)))
        u_works = wage - medical_exp

        u_works = np.where(retirement == 1, u_works - theta[3], u_works)

        u = np.array([u_retires, u_works])

        return u

    true_params = [1, 0.8, -1, 2]

    # Transition functions retired:
    health_transition_retired = np.array(
        [
            [0.2, 0.8],
            [0.1, 0.9]
        ]
    )

    is_retired_transition_retired = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9]
        ]
    )

    pension_transition_retired = np.eye(3)

    medical_exp_transition_retired = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7]
        ]
    )

    age_transition_retired = np.diag(np.ones(shape=29), k=1)
    age_transition_retired[-1, -1] = 1

    # Transition matrices working:
    health_transition_working = np.array(
        [
            [0.4, 0.6],
            [0.1, 0.9]
        ]
    )

    is_retired_transition_working = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9]
        ]
    )

    pension_transition_working = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7]
        ]
    )

    medical_exp_transition_working = np.array(
        [
            [0.5, 0.4, 0.1],
            [0.4, 0.5, 0.1],
            [0.1, 0.5, 0.4]
        ]
    )

    age_transition_working = np.diag(np.ones(shape=29), k=1)
    age_transition_working[-1, -1] = 1

    state_manager = StateManager(
        age=n_age_states,
        health=n_health_states,
        retirement=n_retirement_states,
        pension=n_pension_states,
        medical_exp=n_medical_exp_states,
    )

    transition_matrix_working = StateManager.merge_matrices(age_transition_working,
                                                            health_transition_working,
                                                            is_retired_transition_working,
                                                            pension_transition_working,
                                                            medical_exp_transition_working)

    transition_matrix_retired = StateManager.merge_matrices(age_transition_retired,
                                                            health_transition_retired,
                                                            is_retired_transition_retired,
                                                            pension_transition_retired,
                                                            medical_exp_transition_retired)

    transition_matrix = np.array([transition_matrix_retired, transition_matrix_working])

    # Flaky test
    @average_out(5)
    def test():
        df, ccp = simulate(100, 100, 2, state_manager, true_params, utility_fn, discount_factor, transition_matrix)

        parameter_names = ['const', 'health', 'age', 'work_disutility']

        algorithm = CCP(df['action'].values,
                        df['state'].values,
                        transition_matrix,
                        utility_fn,
                        discount_factor,
                        initial_p=ccp,
                        state_manager=state_manager,
                        parameter_names=parameter_names
                        )

        return algorithm.estimate(start_params=np.random.uniform(size=len(parameter_names)), method='bfgs')

    mean_params = test()

    assert np.abs((mean_params - true_params)/true_params).max() < 0.4
