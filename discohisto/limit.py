"""Tools to assign likelihood limits on signal contributions."""
import os
from dataclasses import asdict, dataclass
from functools import partial

import numpy
import scipy

from . import serial, stats

# levels of:
#     log prob (data | background + signal) - log prob(data | background)
# not only limits, we can also see positive evidence!
# (this will be sign flipped, so limits are set with positive level)
# 0 data => limit is 3 if we select level at 3
DEFAULT_LEVELS = (
    4.5,
    3.0,
    2.0,
    0.5,
    0.0,
    -0.5,
    -2.0,
    -3.0,
    -4.5,
)

# fit processing


def dump_scans(
    label, fit, model_fn, path_limit, ndata, lo, hi, *, nbins=200, print_=False
):
    partial_model = model_fn(fit)
    model_temp = partial_model(ndata)
    predicted_trio = low_central_high(model_temp.prior)

    n_and_suffix = zip(
        (ndata, *predicted_trio), ("observed", "down", "central", "up")
    )

    for n, suffix in n_and_suffix:
        scan_i = scan(partial_model, n, lo, hi, nbins + 1)
        scan_i.dump(path_limit, suffix="_%s_%s" % (label, suffix))

        if scan_i.points[-1]:
            top = "%.1f" % scan_i.points[-1][0]
        else:
            top = "!!!"

        if print_:
            print(f"  {label:>16s} {n=:<6.1f} {top}")


def dump_scan_fit_signal(fit, path_limit, *, print_=False):
    scan_i = scan_fit_signal(fit)
    suffix = "observed"
    label = "signal"
    scan_i.dump(path_limit, suffix="_%s_%s" % (label, suffix))

    if scan_i.points[-1]:
        top = "%.1f" % scan_i.points[-1][0]
    else:
        top = "!!!!"

    if print_:
        print(f"  {label:>16s} n=obs    {top}")


def dump_scan_delta(mu, path_limit, ndata, lo, hi, *, nbins=200, print_=False):
    label = "delta"
    trio = [max(0.0, mu - mu**0.5), mu, mu + mu**0.5]

    n_and_suffix = zip((ndata, *trio), ("observed", "down", "central", "up"))

    for n, suffix in n_and_suffix:
        scan_i = scan_delta(mu, n, lo, hi, nbins + 1)
        scan_i.dump(path_limit, suffix="_%s" % suffix)

        if scan_i.points[-1]:
            top = "%.1f" % scan_i.points[-1][0]
        else:
            top = "!!!"

        if print_:
            print(f"  {label:>16s} {n=:<6.1f} {top}")


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
    levels = list(levels)

    model = partial(partial_model, ndata)
    signals = numpy.linspace(start, stop, num)

    integrals = numpy.array(
        [model(shift=signal).integrate(rtol=rtol) for signal in signals]
    )

    # do no-signal to higher precision to allay error doubts
    integral_zero = model(shift=0.0).integrate(rtol=rtol / 10)
    if start == 0:
        integrals[0] = integral_zero

    # find crosses with log-linear interpolation
    log_ratios = _ratio_log_mean(integrals, integral_zero)
    points = [crosses(signals, log_ratios, value) for value in levels]

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
    """Return a LimitScanFit to assign limits to a FitSignal result.

    Arguments:
        fit: FitSignal-like
        levels: negative log likelihood levels to cross
    """
    if not fit.start == 0:
        raise ValueError(fit.start)
    levels = list(levels)

    signals = numpy.linspace(fit.start, fit.stop, len(fit.levels))

    # these are objective levels (negated log likelihood ratios)
    # so negate with respect to others
    log_ratios = fit.levels[0] - numpy.array(fit.levels)
    points = [crosses(signals, log_ratios, value) for value in levels]

    return LimitScanFit(
        # linspace arguments
        start=fit.start,
        stop=fit.stop,
        # crosses results
        levels=levels,
        points=points,
    )


def scan_delta(
    mu: float,
    ndata: float,
    start: float,
    stop: float,
    num: int,
    *,
    levels: list[float] = DEFAULT_LEVELS,
):
    """Return a LimitScan wih no uncertainty.

    Arguments:
        mu: fixed poisson mean
        ndata: observed data (real data are integer)
        start, stop, num: linspace arguments
        levels: negative log likelihood levels to cross
    """
    levels = list(levels)

    signals = numpy.linspace(start, stop, num)

    log_ratio_0 = stats.poisson_log_minus_max(ndata, mu + 0)
    log_ratios = stats.poisson_log_minus_max(ndata, mu + signals) - log_ratio_0
    points = [crosses(signals, log_ratios, value) for value in levels]

    return LimitScanDelta(
        ndata=ndata,
        # arguments
        start=start,
        stop=stop,
        # crosses results
        levels=levels,
        points=points,
    )


# work functions

standard_normal_cdf = scipy.special.ndtr


def low_central_high(prior, *, x0: float = 1.0, xtol: float = 1e-6):
    """Return crude assignments of -1 sigma, central, and +1 sigma data.

    We integrate to estimate the mean and variance of the data distribution,
    and take the three points as mean +- 1sigma.

    Arguments:
        all passed to quantile(...)
    """
    # spread qs evenly to give equal weights to each
    npoints = 100
    qs = (numpy.arange(npoints) + 0.5) / npoints
    xs = numpy.array([quantile(prior, q, x0=x0, xtol=xtol) for q in qs])

    # data distribution is a weighted mean of poisson distributions
    # variance of each is mean of squares minus square of means
    # poisson is y +- sqrt(y), <n**2> - y**2 = y, so <n**2> = y**2 + y
    # moments add, so get mean over our points
    # some rearrangement later:
    mean = xs.mean()
    std = (xs.var() + mean) ** 0.5

    return max(0.0, mean - std), mean, mean + std


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


def _ratio_log_mean(integrals, integral_zero):
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
    integral_zero: list[float]
    integrals: list[list[float]]

    filename = "scan"

    def dump(self, path, *, suffix):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)


@dataclass(frozen=True)
class LimitScanFit:
    # linspace arguments
    start: float
    stop: float
    # crosses results
    levels: list[float]
    points: list[list[float]]

    filename = "scan_fit"

    def dump(self, path, *, suffix):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)


@dataclass(frozen=True)
class LimitScanDelta:
    ndata: int | float
    # linspace arguments
    start: float
    stop: float
    # crosses results
    levels: list[float]
    points: list[list[float]]

    filename = "scan_delta"

    def dump(self, path, *, suffix):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
