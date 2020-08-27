from functools import reduce
import numpy as np
import pandas as pd

from bidict import frozenbidict


def random_ccp(n_states, n_choices):
    """
    Obtain an array of random choice probabilities.
    :param n_states:
    :param n_choices:
    :return: an array of shape (n_choices, n_states, 1), where the values across choices add up to 1.
    """
    p = np.random.uniform(size=(n_choices, n_states, 1))
    p = (p / p.sum(axis=0))
    return p


class StateManager(object):

    """A convenience class for managing complex state-spaces. The class keeps an internal biyective mapping of state IDs
    and combinations of values for the corresponding state variables.

    :param **kwargs: the keys represent the name of the state variable and the values are the number of possible states.
    """

    def __init__(self, **n_states):
        self.state_names = list(n_states.keys())
        state_sizes = list(n_states.values())

        self.n_states = state_sizes
        self.n_dimensions = len(self.n_states)
        self.total_states = np.prod(self.n_states)

        if self.n_dimensions > 1:
            mesh = self.get_states_mesh().squeeze().T
            mesh = list(map(tuple, mesh))
            index = np.arange(len(mesh))
            self.values_dict = frozenbidict(zip(mesh, index))

    def get_states_mesh(self):
        """

        :return:
        """
        state_dimension = len(self.n_states)
        states_mesh = np.stack(np.meshgrid(*[np.arange(0, n) for n in self.n_states], indexing='ij'), -1)
        return states_mesh.reshape(-1, state_dimension).T.reshape(state_dimension, -1, 1)

    def state_id_for(self, others):
        """Get the unique state ID for a given combination of state variables.

        :param others: can be a list of tuples, a pandas Series or pandas DataFrame, where the values represent the
        value of the state for each of the dimensions used to instantiate the State Manager. Values must be in the same
        as the one used to create the State Manager.
        :return: an array, Series or pandas DataFrame containing the state ID for the values passed.
        """
        if self.n_dimensions == 1:
            return others
        else:
            if type(others) == pd.core.series.Series:
                return others.apply(lambda v: self[v])
            elif type(others) == pd.core.frame.DataFrame:
                df = others.apply(lambda v: self.values_dict[tuple(v)], axis=1)
                return df
            else:
                search_values = map(tuple, others)
                return np.array([self.values_dict[v] for v in search_values])

    def state_variables_for(self, others):
        """Pass a state ID to obtain the original state variables.

        :param others: can be an iterable of `int`, a pandas Series or a pandas DataFrame.
        :return: an array of tuples, Series or pandas DataFrame containing the state ID for the values passed.
        """
        if self.n_dimensions == 1:
            return others
        else:
            if type(others) == pd.core.series.Series:
                return others.apply(lambda v: self.values_dict.inv[v])
            elif type(others) == pd.core.frame.DataFrame:
                if len(others.columns) > 1:
                    raise Exception('The dataframe must have exactly one column.')
                df = others.apply(lambda v: pd.Series(list(self.values_dict.inv[v.values.item()])), axis=1)
                df.columns = self.state_names
                return df
            else:
                return np.array([self.values_dict.inv[v] for v in others])

    def __getitem__(self, x):
        if isinstance(x, int):
            return self.values_dict.inv[x]
        if isinstance(x, tuple):
            return self.values_dict[x]

    @classmethod
    def merge_matrices(cls, *args):
        """Takes several squared matrices and applies recursively the Kronecker product to all of them to obtain a single
        matrix. Useful when you have several transition matrices that you wish to merge into a single one.

        :param args: a list of squared numpy arrays.
        :return: A numpy array with dimensions equal to the multiplication of all the dimensions of the argument
        matrices.s
        """
        return reduce(lambda x, y: np.kron(x, y), args)
