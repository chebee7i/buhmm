buhmm
=====

Bayesian inference for unifilar hidden Markov models in Python.

`buhmm` is an acronym and also a homophone for "bum":

    b = Bayesian
    u = unifilar
    h = hidden
    m = Markov
    m = model

Status: Not ready for general use. Expect major API changes.

Documentation: Eventually.

This package implements the work in:

    Bayesian structural inference for hidden processes
    Christopher C. Strelioff and James P. Crutchfield
    Phys. Rev. E 89, 042119 – Published 10 April 2014
    http://dx.doi.org/10.1103/PhysRevE.89.042119

and extends it in a number of ways. The API was inspired by the reference
implementation written by Christopher C. Strelioff, which is part of `CMPy`,
a Python package for computational mechanics.

One of the primary goals goals for this package was to separate the core
functionality from `CMPy`. Then, `CMPy` could internally provide a its
own compatibility layer. This would enable other libraries and languages
to make use of the inference algorithm, without having to commit to using
`CMPy`. Another goal was to make it fairly low-level and more suitable for
doing many inferences in large `for` loops. This is acheived, in part, through
the use of [NumPy](http://www.numpy.org) arrays and [Cython](http://cython.org).

