import numpy as np
from statsmodels.base.model import GenericLikelihoodModel
from .fixed_point import phi_map, lambda_map
from .probability_tools import random_ccp
from .probability_tools import StateManager
from .simulation_tools import simulate_action_draw, simulate_state_draw


class MaximumLikelihoodDDCModel(GenericLikelihoodModel):
    """
    Abstract interface for all Dynamic Discrete Choice Models that are estimated by Maximum Likelihood.
    The basic components of a DDC model estimated by Maximum Likelihood include:

    - A set of structural parameters of interest: DDCs usually include in the definition of structural parameters
    the conditional choice probabilities, the utility function parameters, the discount factor, the transition
    probabilities and the distribution of the unobservables. This implementation provides tools for estimating the
    utility function parameters and the conditional choice probabilities.
    - The number of choices and states.
    - A transition matrix, defined as a numpy array of shape (n_choices, n_states, n_states). This array represents the
    transition probabilities at each state and its elements are therefore normalized to 1 at the second axis (i.e. the
    operation `transition_matrix.sum(axis=2)` should return an array of ones with shape (n_choices, n_states). Although
    the transition probabilities can be considered structural parameters of interest, the current implementation does
    not estimate them, and therefore must be provided.
    - A utility function: this is a custom function that takes as arguments a list of float structural parameters,
    an array of shape (n_choices, 1, 1) representing the unique values of the choices available to the agent, and a mesh
    of state variables of shape (n_state_dimensions, n_states, 1) representing all the possible combinations of state
    variables.
    - A discount factor: this is a float in the range [0, 1) representing the valuation the agent has of the future.
    Currently this value must be provided, and is therefore not estimated along with the other parameters.
    - Conditional choice probabilities: these are represented as a numpy array of shape (n_choices, n_states, 1),
    representing the agent's probability of making a given choice from the choice set at some state. For some models,
    these probabilities are estimated at the same time as the utility parameters, while in others they must be
    consistently estimated in a first step. This implementation support both cases.


    :param decisions: a single-column vector of decisions data, where choices are represented by consecutive integers
    beginning at zero.
    :param state: an NTx1 vector of state data, where states are represented by consecutive integers beginning at zero
    :param transition_matrix: a numpy array of shape (n_choices, n_states, n_states) containing the transition
    probabilities.
    :param utility_function: a function that takes parameter values, an array of available choices and a mesh of state
    categories as inputs, and returns a real-valued numpy array of shape (n_choices, n_states, 1).
    :param initial_p: the initial value of the conditional choice probabilities as a numpy array of shape
    (n_choices, n_states, 1).
    """

    def __init__(self,
                 decisions,
                 state,
                 transition_matrix,
                 utility_function,
                 discount_factor,
                 initial_p=np.empty(shape=()),
                 parameter_names=None,
                 state_manager=None,
                 **kwds):
        super(MaximumLikelihoodDDCModel, self).__init__(decisions, state, **kwds)
        self.transition_matrix = transition_matrix
        self.n_choices, self.n_states,  _ = transition_matrix.shape
        if state_manager:
            self.state_manager = state_manager
        else:
            self.state_manager = StateManager(state=self.n_states)
        self.utility_function = utility_function
        self.discount_factor = discount_factor
        if initial_p.shape == ():
            self.p = random_ccp(self.n_states, self.n_choices)
        else:
            initial_p[initial_p < 0] = 0
            self.p = initial_p
        self.v = np.random.uniform(size=self.p.shape)
        self.p_convergence_criterion = 10e-10
        self.v_convergence_criterion = 10e-10
        if parameter_names:
            self.data.xnames = parameter_names

    def probability_iteration(self, parameters):
        """Iterates until achieving the fixed point in the probability space.

        :param parameters: the utility function parameters.
        :return: None
        """

        converged = False
        while not converged:
            p_0 = self.p
            self.v = phi_map(self.p,
                             self.transition_matrix,
                             parameters,
                             self.utility_function,
                             self.discount_factor,
                             self.state_manager)
            self.p = lambda_map(self.v)
            delta = np.abs(np.max((self.p - p_0)))
            if delta <= self.p_convergence_criterion:
                converged = True

    def value_iteration(self, parameters):
        """Iterates until achieving the fixed point in the value space.

        :param parameters: the utility function parameters.
        :return: None
        """
        converged = False
        while not converged:
            v_0 = self.v
            self.p = lambda_map(self.v)
            self.v = phi_map(self.p,
                             self.transition_matrix,
                             parameters,
                             self.utility_function,
                             self.discount_factor,
                             self.state_manager)
            delta = np.abs(np.max((self.v - v_0)))
            if delta <= self.v_convergence_criterion:
                converged = True

    def estimate(self, **kwargs):
        """Wraps the fit method from statsmodels `GenericLikelihoodModel`.

        :param kwargs: parameters passed by keyword arguments to the `fit` method of `GenericLikelihoodModel`.
        :return: an instance of `statsmodels.base.model.GenericLikelihoodModelResults` with the estimation results.
        """
        return self.fit(**kwargs)

    def simulate_forward(self, n_draws):
        """This method uses the estimated parameters to simulate the actions of the agent `n_draws` into the future.

        :param n_draws: the number of time steps to simulate forward.
        :return: a numpy array of shape (n_draws + 1, n_obsevations, 2), where the first element in the first axis
        represents the last observed value for the agent, and the rest contain the state and choice values for the
        rest of the time steps.
        """
        history = np.vstack([self.endog.ravel(), self.exog.ravel()]).T.reshape(1, self.endog.shape[0], 2)
        current_history = history
        for _ in range(n_draws):
            next_states = simulate_state_draw(current_history, self.transition_matrix)
            next_choices = simulate_action_draw(self.p, next_states)
            next_history = np.vstack([next_choices, next_states]).T.reshape(1, -1, 2)
            history = np.vstack([history, next_history])
            current_history = next_history
        return history


class NFXP(MaximumLikelihoodDDCModel):
    """
    This class implements the Nested Fixed Point algorithm by Rust (1987). It obtains the fixed point for the
    conditional choice probabilities at each iteration of the maximum likelihood estimation procedure.

    :param decisions: a single-column vector of decisions data, where choices are represented by consecutive integers
    beginning at zero.
    :param state: an NTx1 vector of state data, where states are represented by consecutive integers beginning at zero
    :param transition_matrix: a numpy array of shape (n_choices, n_states, n_states) containing the transition
    probabilities.
    :param utility_function: a function that takes parameter values, an array of available choices and a mesh of state
    categories as inputs, and returns a real-valued numpy array of shape (n_choices, n_states, 1).
    :param initial_p: the initial value of the conditional choice probabilities as a numpy array of shape
    (n_choices, n_states, 1).
    :param parameter_names: a list of `str` containing the names of the target parameters.
    """

    def __init__(self,
                 decisions,
                 state,
                 transition_matrix,
                 utility_function,
                 discount_factor,
                 initial_p=np.empty(shape=()),
                 parameter_names=None,
                 **kwds):
        super(NFXP, self).__init__(decisions,
                                   state,
                                   transition_matrix,
                                   utility_function,
                                   discount_factor,
                                   initial_p,
                                   parameter_names,
                                   **kwds)

    def nloglikeobs(self, parameters):
        """Obtains the likelihood value for the current parameter values by first finding the fixed point for the
        conditional choice probabilities.

        :param parameters: a list of float values.
        :return: a float.
        """
        self.probability_iteration(parameters)
        pr = self.p[self.endog.ravel(), self.exog.ravel()]
        ll = -np.log(pr).sum()
        return ll


class CCP(MaximumLikelihoodDDCModel):
    """
    Implements the Hotz & Miller (1993) Conditional Choice Probability algorithm. The conditional choice probabilities
    must be consistently estimated in a separate step and passed as the `initial_p` argument. This algorithm finds the
    value at each state that is consistent with the passed probabilities.

    :param decisions: a single-column vector of decisions data, where choices are represented by consecutive integers
    beginning at zero.
    :param state: an NTx1 vector of state data, where states are represented by consecutive integers beginning at zero
    :param transition_matrix: a numpy array of shape (n_choices, n_states, n_states) containing the transition
    probabilities.
    :param utility_function: a function that takes parameter values, an array of available choices and a mesh of state
    categories as inputs, and returns a real-valued numpy array of shape (n_choices, n_states, 1).
    :param initial_p: the initial value of the conditional choice probabilities as a numpy array of shape
    (n_choices, n_states, 1). This value is taken as the definitive value of the conditional choice probabilities and
    is therefore not updated any further during the estimation.
    :param parameter_names: a list of `str` containing the names of the target parameters.
    """

    def __init__(self,
                 decisions,
                 state,
                 transition_matrix,
                 utility_function,
                 discount_factor,
                 initial_p,
                 parameter_names=None,
                 **kwds):
        super(CCP, self).__init__(decisions,
                                  state,
                                  transition_matrix,
                                  utility_function,
                                  discount_factor,
                                  initial_p,
                                  parameter_names,
                                  **kwds)

    def nloglikeobs(self, parameters):
        """Updates the value function values and obtains the log-likelihood value consistent with the new value function
        without updating the conditional choice probabilities.

        :param parameters: a list of float values.
        :return: a float.
        """
        self.v = phi_map(self.p,
                         self.transition_matrix,
                         parameters,
                         self.utility_function,
                         self.discount_factor,
                         self.state_manager)
        p = lambda_map(self.v)
        pr = p[self.endog.ravel(), self.exog.ravel()]
        ll = -np.log(pr).sum()
        return ll


class NPL(MaximumLikelihoodDDCModel):
    """Implements the Nested Pseudo-Likelihood algorithm by Aguirregabiria & Mira (2002). Similar to the CCP algorithm,
    but updates both the conditional choice probabilities and the value function. Better performance can be obtained
    if the passed conditional choice probabilities are close to the true parameters.

    :param decisions: a single-column vector of decisions data, where choices are represented by consecutive integers
    beginning at zero.
    :param state: an NTx1 vector of state data, where states are represented by consecutive integers beginning at zero
    :param transition_matrix: a numpy array of shape (n_choices, n_states, n_states) containing the transition
    probabilities.
    :param utility_function: a function that takes parameter values, an array of available choices and a mesh of state
    categories as inputs, and returns a real-valued numpy array of shape (n_choices, n_states, 1).
    :param initial_p: the initial value of the conditional choice probabilities as a numpy array of shape
    (n_choices, n_states, 1).
    :param parameter_names: a list of `str` containing the names of the target parameters.
    :param relaxation_parameter: this is a float value in the range (0, 1] that acts as a learning rate to solve
    convergence issues as proposed in Kasahara & Shimotsu (2008). Values closer to 1 give more preference to new values
    of the conditional choice probabilities, while values closer to 0 give more weight to the values obtained in the
    previous iteration.
    :param npl_maxiter: the maximum allowed number of iterations for the NPL algorithm.
    """

    def __init__(self,
                 decisions,
                 state,
                 transition_matrix,
                 utility_function,
                 discount_factor,
                 initial_p,
                 parameter_names=None,
                 relaxation_param=1.,
                 npl_maxiter=None,
                 **kwds):
        super(NPL, self).__init__(decisions,
                                  state,
                                  transition_matrix,
                                  utility_function,
                                  discount_factor,
                                  initial_p,
                                  parameter_names,
                                  **kwds)
        self.__fit = self.fit
        if 0. <= relaxation_param <= 1.:
            self.relaxation_param = relaxation_param
        else:
            raise Exception('Relaxation param must be a float between 0 and 1.')

        if npl_maxiter is None:
            self.npl_maxiter = npl_maxiter
        elif npl_maxiter > 0:
            self.npl_maxiter = int(npl_maxiter)
        else:
            raise Exception('Max Iterations must be a positive integer')

    def nloglikeobs(self, parameters):
        """Updates the value function and the conditional choice probabilities for each iteration and calculates the
        log-likelihood at the given parameters.

        :param parameters: a list of float values.
        :return: a float.
        """
        self.v = phi_map(self.p,
                         self.transition_matrix,
                         parameters,
                         self.utility_function,
                         self.discount_factor,
                         self.state_manager)
        self.p = (lambda_map(self.v)**self.relaxation_param)*(self.p**(1 - self.relaxation_param))
        pr = self.p[self.endog.ravel(), self.exog.ravel()]
        ll = -np.log(pr).sum()
        return ll

    def estimate(self, **kwargs):
        """Estimate using the NPL algorithm. The procedure performs several CCP updates until convergence or a maximum
        number of iterations.

        :param kwargs: parameters passed by keyword arguments to the `fit` method of `GenericLikelihoodModel`.
        :return: an instance of `statsmodels.base.model.GenericLikelihoodModelResults` with the estimation results.
        """
        results = self.fit(**kwargs)
        converged = False
        n_iterations = 0
        while not converged:
            p_0 = self.p
            results = self.fit(**{**kwargs, 'start_params': results.params})
            delta = np.abs(np.max((self.p - p_0)))
            n_iterations += 1
            if self.npl_maxiter is not None:
                if delta <= self.p_convergence_criterion or n_iterations >= self.npl_maxiter:
                    converged = True
            elif delta <= self.p_convergence_criterion:
                converged = True

        return results
