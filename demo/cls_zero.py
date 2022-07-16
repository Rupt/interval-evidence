"""Serialize test statistic distributions with zero data.

Usage:

python demo/cls_zero.py ${SIGNAL} %{BKG} ${BKG_ERR} ${NSAMPLES}

e.g.

python demo/cls_zero.py 3.0 1.0 0.2 1000

"""
import pickle
import sys

import jax
import numpy
import pyhf

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

pyhf.set_backend("jax")
pyhf.schema.validate = lambda *args, **kwargs: None


def main():
    _, signal_str, bkg_str, err_str, nsamples_str = sys.argv
    signal = float(signal_str)
    bkg = float(bkg_str)
    err = float(err_str)
    nsamples = int(nsamples_str)

    teststat, dist_sb, dist_b = cls_zero(signal, bkg, err, nsamples)

    filename = "_".join(sys.argv).replace(".", "p") + ".pickle"
    with open(filename, "wb") as file_:
        pickle.dump((teststat, dist_sb, dist_b), file_)
    print("wrote %r" % filename)


def cls_zero(signal, bkg, err, nsamples, seed=0):
    numpy.random.seed(seed)

    model = pyhf.simplemodels.uncorrelated_background(
        signal=[signal], bkg=[bkg], bkg_uncertainty=[err]
    )

    data = [0] + model.config.auxdata
    calculator = pyhf.infer.calculators.ToyCalculator(
        data,
        model,
        ntoys=nsamples,
        track_progress=False,
    )

    mu = 1.0
    dist_sb, dist_b = calculator.distributions(mu)
    teststat = calculator.teststatistic(mu)
    return teststat, dist_sb, dist_b


if __name__ == "__main__":
    main()
