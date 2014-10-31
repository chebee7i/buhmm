"""
`buhmm` is a Python implementation of Bayesian inference and model selection
on unifilar hidden Markov models.

The name is an acronym, pronounced as 'bum', which means "Bayesian unifilar
hidden Markov model". Unifilar is a property of hidden Markov models that makes
their topology (with symbols) equal to a deterministic finite automata---from
any current node, there is at most one next node associated with a given
symbol. Recall however, that hidden Markov models are stochastic, so the
deterministic nature of the topology should not be conflated with the
(generally) stochastic nature of its dynamics.

To improve performance, many of the core components of the algorithm are
written in Cython. Hidden Markov models are represented in canonical form,
meaning the nodes and symbols are integers beginning at 0, and the
transition function of the hidden Markov model is represented as dense 2D
NumPy array with shape equal to `(n, k)`, where `n` is the number of nodes and
`k` is the number of symbols in the hidden Markov model.

"""
from .canonical import *
from .counts import *
from .dirichlet import *
