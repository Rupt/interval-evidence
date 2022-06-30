"""Construct partial lebesgue models from fit results."""
from functools import partial

import numpy

import lebesgue


def cabinetry(fit):
    return partial(
        lebesgue.canned.gamma1_trunc_normal,
        mu=fit.yield_pre,
        sigma=fit.error_pre,
    )


def normal(fit):
    return partial(
        lebesgue.canned.gamma1_trunc_normal,
        mu=fit.yield_linear,
        sigma=fit.error_linear,
    )


def normal_log(fit):
    return partial(
        lebesgue.canned.gamma1_log_normal,
        mu=_safe_log(fit.yield_linear),
        sigma=fit.error_log,
    )


def linspace(fit):
    return partial(
        lebesgue.canned.gamma1_regular_linear,
        start=fit.start,
        stop=fit.stop,
        log_rates=numpy.negative(fit.levels),
    )


def mcmc(fit):
    return partial(
        lebesgue.canned.gamma1_regular_uniform,
        start=fit.range_[0],
        stop=fit.range_[1],
        log_rates=_safe_log(fit.yields),
    )


# utility


def _safe_log(x):
    x = numpy.asarray(x)
    iszero = x == 0
    return numpy.where(iszero, -numpy.inf, numpy.log(x + iszero))
