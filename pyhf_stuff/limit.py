"""Tools to assign likelihood limits on signal contributions."""
import os
from dataclasses import asdict, dataclass
from functools import partial

import numpy
import scipy

from . import serial, stats

FILENAME_FORMAT = "limit_%s.json"

DEFAULT_LEVELS = tuple(stats.sigma_to_llr(range(1, 3 + 1)))


def scan(
    partial_model,
    ndata: float,
    start: float,
    stop: float,
    num: int,
    *,
    levels: list[float] = DEFAULT_LEVELS,
    rtol: float = 1e-2,
):
    """Return a LimitScan from a standard linear scan.

    Arguments:
        model: partial model as returned from models.py functions
        ndata: observed data (real data are integer)
        start, stop, num: linspace arguments
        levels: negative log likelihood levels to cross
    """
    model = partial(partial_model, ndata)
    signals = numpy.linspace(start, stop, num)

    integrals = numpy.array(
        [model(shift=signal).integrate(rtol=rtol) for signal in signals]
    )

    # do no-signal to higher precision to allay error doubts
    integral_zero = model(shift=0.0).integrate(rtol=rtol / 10)

    # find crosses with log-linear interpolation
    log_ratios = log_ratio(integrals, integral_zero)

    points = [
        crosses(signals, log_ratios, value) for value in numpy.negative(levels)
    ]

    return LimitScan(
        ndata=ndata,
        # linspace arguments
        start=start,
        stop=stop,
        # scan results
        rtol=rtol,
        integral_zero=list(integral_zero),
        integrals=integrals.tolist(),
        # crosses results
        levels=list(levels),
        points=points,
    )


def log_ratio(integrals, integral_zero):
    bulk = numpy.log(numpy.mean(integrals, axis=1))
    zero = numpy.log(numpy.mean(integral_zero))
    return bulk - zero


# work functions

standard_normal_cdf = scipy.special.ndtr


def quantile(prior, q: float, *, x0: float = 1.0, xtol: float = 1e-6):
    """Return an approximate quantile for the given lebesgue prior.

    Works by interval bisection in its cumulative distribution.

    Written for non-negative priors, so begins from a lower limit of 0.

    Arguments:
        prior: lebesgue Prior-like
        q: quantile (target cumulative distribution function value)
        x0: initial estimated upper bound on the quantile
        xtol: passed to scipy.optimize.bisect
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)

    def func(x):
        return prior.between(0.0, x) - q

    # initial expansion to find an upper bound
    hi = x0
    while not func(hi) > 0:
        hi *= 2

    return scipy.optimize.bisect(func, 0.0, hi, xtol=xtol)


def crosses(x: list[float], y: list[float], value: float) -> list[float]:
    """Return estimates for where the curve x, y crosses value.

    Estimates are linearly interpolated.

    Arguments:
        x: sequence of x-axis coordinates
        y: sequence of y-axis coordinates
        value: y-value we aim to cross
    """
    if not len(x) == len(y):
        raise ValueError((len(x), len(y)))

    results = []
    for x1, x2, y1, y2 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if not min(y1, y2) <= value <= max(y1, y2):
            continue
        # no width special case
        if x1 == x2:
            return float(x1)
        # no height special case: result is undefined
        if y1 == y2:
            return numpy.nan

        # linear interpolation
        x = x1 + (value - y1) * (x2 - x1) / (y2 - y1)
        results.append(float(x))

    return results


# serialization


@dataclass(frozen=True)
class LimitScan:
    ndata: float
    # linspace arguments
    start: float
    stop: float
    # scan results
    rtol: float
    integrals: list[list[float]]
    integral_zero: list[float]
    # crosses results
    levels: list[float]
    points: list[list[float]]

    def dump(self, path, name):
        os.makedirs(path, exist_ok=True)
        serial.dump_json_human(
            asdict(self), os.path.join(path, FILENAME_FORMAT % name)
        )

    @classmethod
    def load(cls, path, name):
        obj_json = serial.load_json(os.path.join(path, FILENAME_FORMAT % name))
        return cls(**obj_json)
