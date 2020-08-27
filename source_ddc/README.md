# Source DDC

### About the library
Source DDC is a library for the simulation and estimation of Dynamic Discrete Choice (DDC) Models.

### Currently Implemented Models
Source DDC provides tools for constructing single-agent DDC models with a multivariate state-space and any number of choices.

Currently implemented estimation algorithms rely on Maximum Likelihood:

- Nested Fixed Point Algorithm (Rust, 1987)
- CCP Estimator (Hotz & Miller, 1993)
- Nested Pseudo-Likelihood Estimator (Aguirregabiria & Mira, 2002)

Areas of future development include the implementation of simulation-based algorithms, dynamic games and improvements that allow the distributed estimation of large-scale models.

## Installation

The current version has been tested on Python 3.6 and 3.7.
To install, clone or download the repository to your local environment and run `pip install .` from the root directory where the `setup.py` file is.
This installs the necessary dependencies.


## Basic Components of a DDC model

A DDC model is made of the following components:

- Agents
- A utility function to be maximized by the agent
- A set of available choices
- The state-space
- The beliefs about the evolution of the state, given in the form of transition matrices
- A discount factor, representing the degree of preference of short-term utility over the long-term

Currently, Source DDC does not support dynamic games. See the **Future Areas of Work** for more.

The model variables are defined in the following way:

```python
n_choices = 2
n_states = 5
discount_factor = 0.95
```

### Defining the state-space using a State Manager

The state space is defined by employing a *State Manager*, which performs operations on the state-space.
When the state-space is multi-dimensional, since it reduces the complexity of managing several state variables and transition matrices.

A State Manager can be instantiated in the following way:

```python
from source_ddc.probability_tools import StateManager
state_manager = StateManager(miles=n_states)
```

### Transition Matrices

The beliefs of the agent about the evolution of the state-space is defined as a numpy array with dimensions n_choices * n_states * n_states.
In other words, you need to specify a number of squared transition matrices equal to the number of choices. See the example below:

```python
import numpy as np
transition_matrix = np.array(
        [
            [
                [1., 0., 0., 0., 0.],
                [0.1, 0.9, 0., 0., 0.],
                [0., 0.1, 0.9, 0., 0.],
                [0., 0., 0.1, 0.9, 0.],
                [0., 0., 0., 0.1, 0.9]
            ],
            [
                [0.4, 0.6, 0., 0., 0.],
                [0.1, 0.3, 0.6, 0., 0.],
                [0., 0.1, 0.3, 0.6, 0.],
                [0., 0., 0.1, 0.3, 0.6],
                [0., 0., 0., 0.1, 0.9]
            ]
        ]
    )
```

### The Utility Function

It is expressed as a python function that receives three arguments: the parameter values, an array of choices and an array of states.
For example: 

```python
    def utility_fn(parameters, choices, states):
        [variable_cost, fixed_cost] = parameters
        m_states, m_actions = np.meshgrid(states, choices)
        return (variable_cost * np.log(m_states + 1) - fixed_cost * m_actions).reshape((len(choices), -1, 1))
```

## Simulating data

Source DDC includes an easy way of performing Monte Carlo simulations.

You can simulate new data in the following way:


```python
from source_ddc.simulation_tools import simulate

df, ccp = simulate(100, 10, n_choices, state_manager, true_params, utility_fn, discount_factor, transition_matrix)
```

This function returns a tuple containing the simulated data and the conditional choice probabilities. 


### Estimating a model

The algorithms are included in the `dynamic` module.

You need to create an instance of the algorithm with the necessary data and use the `estimate` function to perform the estimation:

```python
from source_ddc.algorithms import NFXP
algorithm = NFXP(
    df['action'],
    df['state'],
    transition_matrix,
    utility_fn,
    discount_factor,
    parameter_names=['variable_cost', 'replacement_cost']
    )

result = algorithm.estimate(start_params=[-1, -1], method='bfgs')
```

The `result` object inherits from the Statsmodels [GenericLikelihoodModelResults](https://www.statsmodels.org/stable/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html).

A summary of the results can be obtained with `result.summary()`.

## Testing

Tests run against the installed version. To run them, install the library (in a virtual environment if necessary).

Then, install the necessary dependencies by running `pip install -r requirements.txt` from the root directory.

Tests are found inside the `/test/` directory and are of three kinds:

- Unit tests: employ Monte Carlo simulations to verify that the results returned by the algorithms are as expected.
Run them using `pytest .` from the `/test/unit/` directory.

- Benchmarks: compare the speed of the `estimate` method of each algorithm.
Run them using `pytest .` from the `/test/benchmark/` directory.

- Profiling: measure the time that key parts of the algorithms take when estimating. Helps spot bottlenecks.
Run them using `pytest . --profile` from the `/test/performance/` directory. This will generate profile files which you can visualize using tools like [snakeviz](https://jiffyclub.github.io/snakeviz/)

## Contributing

This project is managed under the principles of: Share, Build, Discuss, Learn.

Feel free to open new Issues, submit Pull Requests, share your use cases and discuss of others.

For the good of the community, remember to keep the conversation polite and productive.


## Maintainers

This project is maintained by researchers at DSOC.

Contact: https://sansan-dsoc.com/contact/

## References

- Aguirregabiria, Victor and Pedro Mira (2002) "Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models." Econometrica 70(4):1519-543.
- Hotz, V. Joseph, and Robert A. Miller. "Conditional Choice Probabilities and the Estimation of Dynamic Models." The Review of Economic Studies 60, no. 3 (1993): 497-529.
- Kasahara, Hiroyuki, and Katsumi Shimotsu. "Sequential Estimation of Structural Models With a Fixed Point Constraint." Econometrica 80, no. 5 (2012): 2303-319. Accessed August 21, 2020. http://www.jstor.org/stable/23271448.
- Rust, John (1987) "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher." Econometrica, 55: 999–1033.
