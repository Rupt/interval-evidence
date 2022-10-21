"""Check in all examples ordering in data matches ordering in limits.

Usage:

python report/check_orderings.py

"""
import numpy

from report import frame


def main():
    frame_ = frame.load("report/results.csv")
    # also check our copy of reported results is accurate
    assert numpy.array_equal(frame_.reported_n, frame_.region_n)
    print_orderings(frame_, "cabinetry")
    print_orderings(frame_, "normal")
    print_orderings(frame_, "normal_log")
    print_orderings(frame_, "linspace")
    print_orderings(frame_, "delta")
    print_orderings(frame_, "mcmc")


def print_orderings(frame_, label):
    search_ = frame_.search_
    region_ = frame_.region_
    nobs = frame_.reported_n
    nexp = getattr(frame_, f"limit_{label}_nexp")
    nexp_hi = getattr(frame_, f"limit_{label}_nexp_hi")
    nexp_lo = getattr(frame_, f"limit_{label}_nexp_lo")
    obs = getattr(frame_, f"limit_{label}_3obs")
    exp = getattr(frame_, f"limit_{label}_3exp")
    exp_hi = getattr(frame_, f"limit_{label}_3exp_hi")
    exp_lo = getattr(frame_, f"limit_{label}_3exp_lo")

    parts = zip(
        search_,
        region_,
        nobs,
        nexp,
        nexp_hi,
        nexp_lo,
        obs,
        exp,
        exp_hi,
        exp_lo,
    )

    any_ = False

    for items in parts:
        (
            search_i,
            region_i,
            nobs_i,
            nexp_i,
            nexp_hi_i,
            nexp_lo_i,
            obs_i,
            exp_i,
            exp_hi_i,
            exp_lo_i,
        ) = items

        # central
        excess_data = nobs_i > nexp_i
        excess_limit = obs_i > exp_i

        if excess_data != excess_limit:
            any_ = True
            print(
                "%28s %28s %6d %6.1f %6.1f %6.1f"
                % (search_i, region_i, nobs_i, nexp_i, obs_i, exp_i)
            )

        # hi
        excess_data = nobs_i > nexp_hi_i
        excess_limit = obs_i > exp_hi_i

        if excess_data != excess_limit:
            any_ = True
            print(
                "hi %28s %28s %6d %6.1f %6.1f %6.1f"
                % (search_i, region_i, nobs_i, nexp_hi_i, obs_i, exp_hi_i)
            )

        # lo
        excess_data = nobs_i > nexp_lo_i
        excess_limit = obs_i > exp_lo_i

        if excess_data != excess_limit:
            any_ = True
            print(
                "lo %28s %28s %6d %6.1f %6.1f %6.1f"
                % (search_i, region_i, nobs_i, nexp_lo_i, obs_i, exp_lo_i)
            )

    if not any_:
        print("ALL OK %r" % label)


if __name__ == "__main__":
    main()
