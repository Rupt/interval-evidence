import numpy

import lebesgue

from .limit import quantile, standard_normal_cdf


def test_quantile():
    # go far from the truncation
    mu = 100
    sigma = 1
    prior = lebesgue.canned.gamma1_trunc_normal(0, mu, sigma).prior

    sigmas = [-1, 0, 1]
    qs = [quantile(prior, q_i) for q_i in standard_normal_cdf([-1, 0, 1])]

    numpy.testing.assert_allclose(qs, mu + sigma * numpy.array(sigmas))
