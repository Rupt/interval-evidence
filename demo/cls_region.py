"""Serialize test statistic distributions for a region @ 3 signal.

Usage:

python demo/cls_region.py ${NSAMPLES} ${REGIONPATH}

e.g.

python demo/cls_region.py 100 \
searches/atlas_susy_3Lresonance_2020/SR3l_90_110_all/

"""
import pickle
import sys

import numpy
import pyhf

from discohisto import region


def main():
    _, nsamples, region_path = sys.argv
    nsamples = int(nsamples)

    teststat, dist_sb, dist_b = cls_region(nsamples, region_path)

    filename = "cls_region_" + region_path.replace("/", "_")
    if not filename.endswith("_"):
        filename += "_"
    filename += "%d.pickle" % nsamples
    with open(filename, "wb") as file_:
        pickle.dump((teststat, dist_sb, dist_b), file_)
    print("wrote %r" % filename)


def cls_region(nsamples, region_path, seed=0):
    numpy.random.seed(seed)

    reg = region.Region.load(region_path)

    workspace = region.discovery_workspace(reg)

    model = workspace.model()
    data = workspace.data(model)
    signal = 3.0

    calculator = pyhf.infer.calculators.ToyCalculator(
        data,
        model,
        ntoys=nsamples,
        track_progress=False,
    )

    dist_sb, dist_b = calculator.distributions(signal)
    teststat = calculator.teststatistic(signal)
    return teststat, dist_sb, dist_b


if __name__ == "__main__":
    main()
