from test import test_funcs

from . import _test_blind


def run_tests():

    tests = [
        _test_blind.test_simple_model_blind,
    ]

    test_funcs(tests)