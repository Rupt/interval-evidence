"""
Define functions for benchmarking.
"""

from lebesgue import Model
from lebesgue.likelihood import poisson
from lebesgue.prior import log_normal, plus


def model_integrate_1():
    return Model(poisson(3), plus(1, log_normal(10, 1))).integrate()
