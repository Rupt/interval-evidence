"""Tools to assign likelihood limits on signal contributions."""
import os
from dataclasses import asdict, dataclass
from functools import partial

import numpy
import scipy

from . import serial, stats

DEFAULT_LEVELS = (3.0,) + tuple(stats.sigma_to_llr([1, 2, 3]))

# for fit processing


def dump_scans(label, fit, model_fn, path_limit, ndata, lo, hi, *, nbins=200):
    partial_model = model_fn(fit)
    model_temp = partial_model(ndata)

    predicted_trio = low_central_high(model_temp.prior)

    n_and_suffix = zip(
        (ndata, *predicted_trio), ("observed", "down", "central", "up")
    )

    for n, suffix in n_and_suffix:
        scan_i = scan(partial_model, n, lo, hi, nbins + 1)
        scan_i.dump(path_limit, suffix="_%s_%s" % (label, suffix))


def dump_scan_fit_signal(label, fit, path_limit):
    scan_fit_signal_i = scan_fit_signal(fit)
    suffix = "observed"
    scan_fit_signal_i.dump(path_limit, suffix="_%s_%s" % (label, suffix))


# core


def scan(
    partial_model,
    ndata: float,
    start: float,
    stop: float,
    num: int,
    *,
    levels: list[float] = DEFAULT_LEVELS,
    rtol: float = 1e-3,
):
    """Return a LimitScan from a standard linear scan.

    Arguments:
        model: partial model as returned from models.py functions
        ndata: observed data (real data are integer)
        start, stop, num: linspace arguments
        levels: negative log likelihood levels to cross
        rtol: relative tolerance argument to Model.integrate
    """
    model = partial(partial_model, ndata)
    signals = numpy.linspace(start, stop, num)

    integrals = numpy.array(
        [model(shift=signal).integrate(rtol=rtol) for signal in signals]
    )

    # do no-signal to higher precision to allay error doubts
    integral_zero = model(shift=0.0).integrate(rtol=rtol / 10)

    # find crosses with log-linear interpolation
    levels = list(levels)
    log_ratios = log_ratio(integrals, integral_zero)

    points = [
        crosses(signals, log_ratios, value) for value in numpy.negative(levels)
    ]

    return LimitScan(
        ndata=ndata,
        # arguments
        start=start,
        stop=stop,
        rtol=rtol,
        # crosses results
        levels=levels,
        points=points,
        # scan results
        integral_zero=list(integral_zero),
        integrals=integrals.tolist(),
    )


def scan_fit_signal(fit, *, levels: list[float] = DEFAULT_LEVELS):
    """Return a LimitFitSignal to assign limits to a FitSignal result.

    Arguments:
        fit: FitSignal-like
        levels: negative log likelihood levels to cross
    """
    signals = numpy.linspace(fit.start, fit.stop, len(fit.levels))

    levels = list(levels)
    log_ratio = numpy.array(fit.levels) - min(fit.levels)

    points = [crosses(signals, log_ratio, value) for value in levels]

    return LimitFitSignal(
        # linspace arguments
        start=fit.start,
        stop=fit.stop,
        # crosses results
        levels=levels,
        points=points,
    )


# work functions

standard_normal_cdf = scipy.special.ndtr


def low_central_high(prior, *, x0: float = 1.0, xtol: float = 1e-6):
    """Return crude assignments of -1 sigma, central, and +1 sigma data.

    The assignemtn is to take quantiles of the prior, then take their
    mean data shifted down or up by its square root for the variations.

    This is not, to my knowledge, a meaningful median (or any other standard
    statistic), but it is simple and converges sensibly for high precision
    and mean >> 1.

    Arguments:
        all passed to quantile(...)
    """
    q0, q1, q2 = [quantile(prior, q) for q in standard_normal_cdf([-1, 0, 1])]

    return max(0.0, q0 - q0**0.5), q1, q2 + q2**0.5


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


def log_ratio(integrals, integral_zero):
    bulk = numpy.log(numpy.mean(integrals, axis=1))
    zero = numpy.log(numpy.mean(integral_zero))
    return bulk - zero


# serialization


@dataclass(frozen=True)
class LimitScan:
    ndata: int | float
    # arguments
    start: float
    stop: float
    rtol: float
    # crosses results
    levels: list[float]
    points: list[list[float]]
    # scan results
    integrals: list[list[float]]
    integral_zero: list[float]

    filename = "scan"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)


@dataclass(frozen=True)
class LimitFitSignal:
    # linspace arguments
    start: float
    stop: float
    # crosses results
    levels: list[float]
    points: list[list[float]]

    filename = "scan_fit_signal"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
