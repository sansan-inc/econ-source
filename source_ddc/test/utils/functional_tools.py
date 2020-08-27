import functools
import numpy as np


def average_out(n_repetitions):
    """Takes a function that returns a flat array of parameters, runs it `n_repetitions` times and returns the mean
    parameter values. Useful for testing using Monte Carlo simulations to prevent flaky tests due to sampling error.
    :param n_repetitions: how many times to sample from the parameter space.
    :return: a flat numpy array of float values representing the mean parameters.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return np.vstack([fn(*args, **kwargs).params for _ in range(n_repetitions)]).mean(axis=0)
        return wrapper
    return decorator
