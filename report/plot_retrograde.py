"""Dump a plot to show that limits can move backwards relative to data.

Usage:

python report/plot_retrograde.py

"""
import numpy
import scipy.special
from matplotlib import pyplot

from discohisto.limit import crosses
from discohisto.stats import poisson_log_minus_max
from report import lib_plot


def main():
    lib_plot.default_init()
    plot_r(
        [11, 12, 13],
        {0: 1, 20: 99},
        [0, 24],
    )


def plot_r(ns, point_to_weight, range_):
    x = numpy.linspace(*range_, 513)

    points = list(point_to_weight)
    weights = numpy.array(list(point_to_weight.values()))
    log_weights = numpy.log(weights / weights.sum())

    def logl(n, x):
        return _logmeanexp(
            [
                poisson_log_minus_max(n, x + point) + log_weight
                for point, log_weight in zip(points, log_weights)
            ]
        )

    # TODO: use standard colors from lib_plot
    colors = ["xkcd:blue", "xkcd:red", "xkcd:orange"]

    figure, axis = pyplot.subplots(
        figsize=(lib_plot.WIDTH_COLUMN, lib_plot.WIDTH_COLUMN * 0.98),
        dpi=400,
        gridspec_kw={
            "top": 0.999,
            "right": 0.999,
            "bottom": 0.10,
            "left": 0.13,
        },
    )

    lines = []
    for n_i, color_i in zip(ns, colors):
        logr = logl(n_i, x) - logl(n_i, 0)
        (line,) = axis.plot(x, logr, color_i)
        lines.append(line)

        cross = crosses(x, logr, -3)[-1]
        print(f"{n_i = }: {cross = } ({cross:.1f})")

        axis.arrow(
            x=cross,
            dx=0,
            y=-4.5,
            dy=0.6,
            linewidth=1,
            head_width=0.5,
            head_length=0.2,
            overhang=0.1,
            color=color_i,
            length_includes_head=True,
        )

    axis.axhline(-3, c="k", ls="--", linewidth=0.6, zorder=0.5)

    axis.text(
        0.03,
        0.99,
        r"Prior: $\propto \delta(x-0) + 99\,\delta(x-20)$",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    axis.legend(
        lines,
        [rf"$n={n_i}$" for n_i in ns],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.71),
        framealpha=0,
        handletextpad=0.4,
        borderpad=0,
        borderaxespad=0,
    )

    axis.set_ylim(-5.9, 0.1)
    axis.set_xlim(*range_)
    axis.set_xlabel(r"$s$")
    axis.set_ylabel(r"$\log R$")
    axis.xaxis.set_label_coords(0.5, -0.06)
    axis.yaxis.set_label_coords(-0.09, 0.5)
    axis.spines.top.set_visible(False)
    axis.spines.right.set_visible(False)

    axis.tick_params(which="major", direction="in", length=5)
    axis.tick_params(which="minor", direction="in", length=3)

    outpath = "retrograde"
    outpath += "_n" + "_".join(map(str, ns))
    outpath += "_d" + "_".join(map(str, point_to_weight.keys()))
    outpath += "_w" + "_".join(map(str, point_to_weight.values()))
    outpath += ".pdf"
    figure.savefig(outpath)
    print("wrote %r" % outpath)
    pyplot.close(figure)


def _logmeanexp(x, axis=0):
    x = numpy.asarray(x)
    norm = numpy.log(x.shape[axis])
    return scipy.special.logsumexp(x, axis=axis) - norm


if __name__ == "__main__":
    main()
