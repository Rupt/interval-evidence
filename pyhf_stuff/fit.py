"""Extract staistics from pyhf models."""
import warnings
from contextlib import contextmanager

import numpy
import scipy

from .region_properties import region_properties

# TODO split and import

# expansion for Gaussian approximation


def normal(region):
    state = region_properties(region)

    optimum = scipy.optimize.minimize(
        state.objective_value_and_grad,
        state.init,
        bounds=state.bounds,
        jac=True,
        method="L-BFGS-B",
    )

    # if gaussian, then covariance is inverse hessian
    cov = state.objective_hess_inv(optimum.x)

    # approximate signal region yield (log) linearly from gradients
    yield_value, yield_grad = state.yield_value_and_grad(optimum.x)

    yield_std = _quadratic_form(cov, yield_grad) ** 0.5

    # d/dx log(f(x)) = f'(x) / f(x)
    yield_log_std = _quadratic_form(cov, yield_grad / yield_value) ** 0.5

    # values are 0-dim jax arrays; cast to floats
    return {
        "linear": {
            "yield": float(yield_value),
            "error": float(yield_std),
        },
        "log": {
            "yield": float(numpy.log(yield_value)),
            "error": float(yield_log_std),
        },
    }


def _quadratic_form(matrix, vector):
    return matrix.dot(vector).dot(vector)


# levels of fixed values


def interval(region, *, levels=(0.5, 2, 4.5)):
    state = region_properties(region)

    optimum = scipy.optimize.minimize(
        state.objective_value_and_grad,
        state.init,
        bounds=state.bounds,
        jac=True,
        method="L-BFGS-B",
    )

    def minmax_given_level(level):
        constaint = scipy.optimize.NonlinearConstraint(
            state.objective_value,
            optimum.fun + level,
            optimum.fun + level,
            jac=state.objective_grad,
        )

        with _suppress_bounds_warning():
            minimum = scipy.optimize.minimize(
                state.yield_value_and_grad,
                state.init,
                bounds=state.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
            )

        def negative_yield_value_and_grad(x):
            value, grad = state.yield_value_and_grad(x)
            return -value, -grad

        with _suppress_bounds_warning():
            maximum = scipy.optimize.minimize(
                negative_yield_value_and_grad,
                state.init,
                bounds=state.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
            )

        return minimum.fun, -maximum.fun

    levels = list(levels)

    intervals = [minmax_given_level(level) for level in levels]

    return {
        "levels": levels,
        "intervals": intervals,
    }


@contextmanager
def _suppress_bounds_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            (
                "Values in x were outside bounds during a "
                "minimize step, clipping to bounds"
            ),
            category=RuntimeWarning,
        )
        yield


# linear scan, like numpy.linspace


def linspace(region, start, stop, num):
    state = region_properties(region)

    def optimum_given_yield(yield_):
        constaint = scipy.optimize.NonlinearConstraint(
            state.yield_value,
            yield_,
            yield_,
            jac=state.yield_grad,
        )

        optimum = scipy.optimize.minimize(
            state.objective_value_and_grad,
            state.init,
            bounds=state.bounds,
            jac=True,
            method="SLSQP",
            constraints=constaint,
        )

        return optimum.fun

    levels = [
        optimum_given_yield(yield_)
        for yield_ in numpy.linspace(start, stop, num)
    ]

    return {
        "start": start,
        "stop": stop,
        "levels": levels,
    }


# Markov Chain Monte Carlo

# utilities


def filename(func):
    return func.__name__ + ".json"
