"""Tools to assign likelihood limits on signal contributions."""
import numpy
import scipy

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
            return x1
        # no height special case: result is undefined
        if y1 == y2:
            return numpy.nan

        # linear interpolation
        x = x1 + (value - y1) * (x2 - x1) / (y2 - y1)
        results.append(x)

    return results
