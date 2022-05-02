import sys
import traceback

import lebesgue
import lebesgue._test_prior_lognormal
import lebesgue._test_prior_plus
import lebesgue._test_quad_bound


def main():

    tests = [
        lebesgue._test_quad_bound.test_fpow,
        lebesgue._test_prior_lognormal.test_gaussian_dcdf,
        lebesgue._test_prior_lognormal.test_between,
        lebesgue._test_prior_lognormal.test_args,
        lebesgue._test_prior_plus.test_shift,
        lebesgue._test_prior_plus.test_args,
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
