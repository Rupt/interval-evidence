"""Statistical stuff"""
import numpy
import scipy


def sigma_to_llr(sigma):
    return numpy.copysign(
        0.5 * numpy.power(sigma, 2),
        sigma,
    )


def llr_to_sigma(llr):
    return numpy.copysign(
        numpy.sqrt(2 * numpy.abs(llr)),
        llr,
    )


def poisson_log_minus_max(n, mu):
    # log(e^-x x^n / n!) - log(e^-n n^n / n!)
    # = -x + nlogx - (-n + nlogn)
    # in convention 0log0 -> 0
    # maximum(n, 1) just avoids a div0 error
    return n - mu + scipy.special.xlogy(n, mu / numpy.maximum(n, 1))
