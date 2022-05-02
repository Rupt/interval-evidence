import sys
import traceback

import lebesgue
import lebesgue._test_prior_lognormal
import lebesgue._test_quad_bound


def main():

    tests = [
        lebesgue._test_quad_bound.test_fpow,
        lebesgue._test_prior_lognormal.test_gaussian_dcdf,
        lebesgue._test_prior_lognormal.test_log_normal_between,
        lebesgue._test_prior_lognormal.test_log_normal_arguments,
    ]

    for test in tests:
        try:
            test()
            print(end=".", flush=True)
        except AssertionError:
            print(end="x", flush=True)
            print(file=sys.stderr)
            traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
