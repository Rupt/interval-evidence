"""
Define functions for benchmarking.
"""
from lebesgue.canned import poisson_log_normal


def model_integrate_1():
    return poisson_log_normal(3, 10.0, 1.0, shift=1).integrate()
