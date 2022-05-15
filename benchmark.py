"""
Define functions for benchmarking.
"""
from lebesgue.canned import poisson_log_normal


def model_integrate_1():
    poisson_log_normal(0, 0.0, 1.0, shift=1).integrate()
    poisson_log_normal(30, 10.0, 1.0, shift=1).integrate()
    poisson_log_normal(3, 20.0, 1.0, shift=1).integrate()
