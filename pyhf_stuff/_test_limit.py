import numpy

import lebesgue

from .limit import crosses, quantile, standard_normal_cdf


def test_quantile():
    # go far from the truncation to avoid its effects
    mu = 10
    sigma = 1
    prior = lebesgue.canned.gamma1_trunc_normal(0, mu, sigma).prior

    sigmas = [-1, 0, 1]
    qs = [quantile(prior, q_i) for q_i in standard_normal_cdf([-1, 0, 1])]

    numpy.testing.assert_allclose(qs, mu + sigma * numpy.array(sigmas))


def test_crosses():
    x = [0, 1, 2, 4]
    y = [0, 1, 0, 2]

    chk = crosses(x, y, 0.2)
    ref = [0.2, 1.8, 2.2]
    numpy.testing.assert_array_equal(chk, ref)
