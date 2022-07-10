"""Statistical stuff"""
import numpy


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
