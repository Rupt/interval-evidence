import sys
import traceback

# import lebesgue
# import lebesgue._test_bayes
import lebesgue._test_cephes_ndtr
import lebesgue._test_likelihood_poisson

# import lebesgue._test_prior_log_normal
# import lebesgue._test_prior_plus
import lebesgue._test_quad_bound


def main():

    tests = [
        # b
        # lebesgue._test_bayes.test_args_likelihood,
        # lebesgue._test_bayes.test_args_prior,
        # lebesgue._test_bayes.test_args_model,
        # lebesgue._test_bayes.test_monotonic,
        # c
        lebesgue._test_cephes_ndtr.test_ndtr,
        lebesgue._test_cephes_ndtr.test_signatures,
        # l
        lebesgue._test_likelihood_poisson.test_args,
        lebesgue._test_likelihood_poisson.test_poisson_interval,
        lebesgue._test_likelihood_poisson.test_invg_lo,
        lebesgue._test_likelihood_poisson.test_invg_hi,
        lebesgue._test_likelihood_poisson.test_signatures,
        # p
        # lebesgue._test_prior_log_normal.test_gaussian_dcdf,
        # lebesgue._test_prior_log_normal.test_between,
        # lebesgue._test_prior_log_normal.test_args,
        # lebesgue._test_prior_plus.test_shift,
        # lebesgue._test_prior_plus.test_args,
        # q
        lebesgue._test_quad_bound.test_fpow,
        # lebesgue._test_quad_bound.test_model,
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
