import sys
import traceback

import lebesgue
import lebesgue._test_bayes
import lebesgue._test_canned
import lebesgue._test_cephes_ndtr
import lebesgue._test_likelihood_poisson
import lebesgue._test_prior_add
import lebesgue._test_prior_normal
import lebesgue._test_prior_trunc
import lebesgue._test_quad_bound


def main():

    tests = [
        # b
        lebesgue._test_bayes.test_args_likelihood,
        lebesgue._test_bayes.test_args_prior,
        lebesgue._test_bayes.test_args_model,
        lebesgue._test_bayes.test_monotonic,
        lebesgue._test_bayes.test_model_mass,
        # c
        lebesgue._test_canned.test_poisson_log_normal,
        lebesgue._test_canned.test_poisson_trunc_normal,
        lebesgue._test_cephes_ndtr.test_ndtr,
        # l
        lebesgue._test_likelihood_poisson.test_args,
        lebesgue._test_likelihood_poisson.test_poisson_interval,
        lebesgue._test_likelihood_poisson.test_invg_lo,
        lebesgue._test_likelihood_poisson.test_invg_hi,
        # p
        lebesgue._test_prior_normal.test_gaussian_dcdf,
        lebesgue._test_prior_normal.test_between,
        lebesgue._test_prior_normal.test_args,
        lebesgue._test_prior_add.test_shift,
        lebesgue._test_prior_add.test_args,
        lebesgue._test_prior_trunc.test_values,
        lebesgue._test_prior_trunc.test_args,
        # q
        lebesgue._test_quad_bound.test_fpow,
    ]

    for test in tests:
        try:
            test()
            print(end=".", flush=True)
        except AssertionError:
            print(end="!", flush=True)
            print(file=sys.stderr)
            traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
