from test import test_funcs

from . import (
    _test_bayes,
    _test_cache,
    _test_canned,
    _test_cephes_ndtr,
    _test_invg,
    _test_likelihood_normal,
    _test_likelihood_poisson,
    _test_prior_add,
    _test_prior_mixture,
    _test_prior_normal,
    _test_prior_rpwlet,
    _test_prior_trunc,
    _test_quad_bound,
)


def run_tests():

    tests = [
        # b
        _test_bayes.test_args_likelihood,
        _test_bayes.test_args_prior,
        _test_bayes.test_args_model,
        _test_bayes.test_monotonic,
        _test_bayes.test_model_mass,
        # c
        _test_cache.test_caching,
        _test_cache.test_put,
        _test_canned.test_poisson_log_normal,
        _test_canned.test_poisson_trunc_normal,
        _test_canned.test_same_funcs,
        _test_cephes_ndtr.test_ndtr,
        # i
        _test_invg.test_invg_lo,
        _test_invg.test_invg_hi,
        _test_invg.test_ginterval,
        # l
        _test_likelihood_normal.test_args,
        _test_likelihood_normal.test_interval,
        _test_likelihood_poisson.test_args,
        _test_likelihood_poisson.test_interval,
        _test_likelihood_poisson.test_gamma1_args,
        _test_likelihood_poisson.test_gamma1_interval,
        # p
        _test_prior_normal.test_gaussian_dcdf,
        _test_prior_normal.test_between,
        _test_prior_normal.test_args,
        _test_prior_add.test_shift,
        _test_prior_add.test_args,
        _test_prior_trunc.test_values,
        _test_prior_trunc.test_args,
        _test_prior_mixture.test_values_pair,
        _test_prior_mixture.test_values_trio,
        _test_prior_mixture.test_args,
        _test_prior_rpwlet.test_first_example,
        _test_prior_rpwlet.test_values,
        _test_prior_rpwlet.test_args,
        # q
        _test_quad_bound.test_fpow,
        _test_quad_bound.test_normal,
        _test_quad_bound.test_next_pow2,
    ]

    test_funcs(tests)
