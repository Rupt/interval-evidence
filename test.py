import sys
import traceback

import lebesgue
import lebesgue._test_quad_bound


def main():

    tests = [
        lebesgue._test_quad_bound.test,
        lebesgue._test_quad_bound.test,
        lebesgue._test_quad_bound.test,
    ]

    for test in tests:
        try:
            test()
            print(end=".")
        except AssertionError:
            print(end="x", flush=True)
            print(file=sys.stderr)
            traceback.print_exc(limit=-3)

    print()


if __name__ == "__main__":
    main()
