from test import test_funcs

from . import _test_blind, _test_stats


def run_tests():

    tests = [
        _test_blind.test_simple_model_blind,
        _test_blind.test_model,
        _test_stats.test_sigma_to_fro_llr,
    ]

    test_funcs(tests)
