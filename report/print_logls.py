"""Print information about s=0 likelihood ratios.

Usage:

python report/print_logls.py

"""
import numpy
import scipy.optimize
import scipy.special

from report import frame


def main():
    frame_ = frame.load("report/results.csv")

    print("mean logls")
    print_mean_logls(frame_)
    print()
    print("mixtures")
    print_optimized_mixture(frame_)


def print_mean_logls(frame_):
    name_to_mean_logl = {
        "cabinetry": frame_.limit_cabinetry_logl.mean(),
        "normal": frame_.limit_normal_logl.mean(),
        "normal_log": frame_.limit_normal_log_logl.mean(),
        "delta": frame_.limit_delta_logl.mean(),
        "linspace": frame_.limit_linspace_logl.mean(),
        "mcmc": frame_.limit_mcmc_logl.mean(),
    }

    ref = max(name_to_mean_logl.values())

    for name, q in name_to_mean_logl.items():
        print("%15s %7.4f %7.4f" % (name, q, q - ref))


def print_optimized_mixture(frame_):
    name_to_mixture_part = {
        "cabinetry": frame_.limit_cabinetry_logl,
        "normal_log": frame_.limit_normal_log_logl,
        "linspace": frame_.limit_linspace_logl,
        "mcmc": frame_.limit_mcmc_logl,
    }

    parts = numpy.stack(list(name_to_mixture_part.values())).T

    def mixture_mean_logl(x):
        log_weights = _log_softmax(x)
        return scipy.special.logsumexp(parts + log_weights, axis=1).mean()

    # logit coordinates have a shift freedom. Constrain it by setting x[-1]=0
    def loss(x_start):
        x = numpy.append(x_start, 0.0)
        return -mixture_mean_logl(x)

    result = scipy.optimize.minimize(
        loss, [0.0] * (len(name_to_mixture_part) - 1)
    )
    print("fit result:")
    print(result)
    print()

    result_weights = numpy.exp(_log_softmax(numpy.append(result.x, 0.0)))
    print("best fit weights:", result_weights)
    print()

    print("mixture mean logls")
    print("%15s %7.4f _______" % ("mixture", -loss(result.x)))
    x_p6_p4 = _safe_log([0.6, 0.4, 0])
    # offset to wash out the appended zero
    print("%15s %7.4f _______" % (".6, .4", -loss(x_p6_p4 + 300)))


def _log_softmax(x):
    # log(e^xi / sum e^xi)
    s = x - x.max()
    return s - numpy.log(numpy.exp(s).sum())


def _safe_log(x):
    x = numpy.asarray(x)
    iszero = x == 0
    return numpy.where(iszero, -numpy.inf, numpy.log(x + iszero))


if __name__ == "__main__":
    main()
