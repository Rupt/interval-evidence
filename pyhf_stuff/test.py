from test import test_funcs

from . import _test_blind, _test_limit, _test_region_properties, _test_stats


def run_tests():

    tests = [
        _test_blind.test_simple_model_blind,
        _test_blind.test_model,
        _test_stats.test_sigma_to_fro_llr,
        _test_limit.test_quantile,
        _test_limit.test_crosses,
        _test_region_properties.test_cache,
    ]

    test_funcs(tests)
