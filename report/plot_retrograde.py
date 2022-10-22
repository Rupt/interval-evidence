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


# TODO: use standard colors from lib_plot
COLORS = ["xkcd:blue", "xkcd:red", "xkcd:orange"]

def main():
    lib_plot.default_init()
    ns = [11, 12, 13]
    point_to_weight = {0: 1, 20: 99}
    cross3 = plot_retrograde(ns, point_to_weight, range_=[0, 24])

    plot_models(cross3, ns, point_to_weight, range_=[0, 44])


def plot_retrograde(ns, point_to_weight, range_):
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

    figure, axis = pyplot.subplots(
        figsize=(lib_plot.WIDTH_COLUMN, lib_plot.WIDTH_COLUMN * 0.62),
        dpi=400,
        gridspec_kw={
            "top": 0.99,
            "right": 0.999,
            "bottom": 0.15,
            "left": 0.13,
        },
    )

    lines = []
    cross3 = []
    x = numpy.linspace(*range_, 513)
    for n_i, color_i in zip(ns, COLORS):
        logr = logl(n_i, x) - logl(n_i, 0)
        (line,) = axis.plot(x, logr, color_i)
        lines.append(line)

        cross = crosses(x, logr, -3)[-1]
        print(f"{n_i = }: {cross = } ({cross:.1f})")
        cross3.append(cross)

        axis.arrow(
            x=cross,
            dx=0,
            y=-4.5,
            dy=0.6,
            linewidth=1,
            head_width=0.4,
            head_length=0.2,
            overhang=0.1,
            color=color_i,
            length_includes_head=True,
        )

    axis.axhline(-3, c="k", ls="--", linewidth=0.6, zorder=0.5)

    axis.text(
        0.06,
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
        loc="lower right",
        bbox_to_anchor=(0.98, 0.52),
        framealpha=0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderpad=0,
        borderaxespad=0,
    )

    axis.set_xlim(*range_)
    axis.set_ylim(-5.9, 0.1)
    axis.set_yticks(range(0, -6, -1))
    axis.set_xlabel(r"signal")
    axis.set_ylabel(r"$\log R$")
    axis.xaxis.set_label_coords(0.5, -0.09)
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
    return cross3


def plot_models(signals, ns, point_to_weight, range_):
    ylim = (-5.9, 0.1)

    figure, axis = pyplot.subplots(
        figsize=(lib_plot.WIDTH_COLUMN, lib_plot.WIDTH_COLUMN * 0.62),
        dpi=400,
        gridspec_kw={
            "top": 0.988,
            "right": 0.999,
            "bottom": 0.15,
            "left": 0.13,
        },
    )

    # plot likelihoods
    x = numpy.linspace(*range_, 513)
    for i, (n_i, color_i) in enumerate(zip(ns, COLORS)):
        logy = poisson_log_minus_max(n_i, x)
        axis.plot(
            x,
            logy,
            "k",
            lw=0.8,
            linestyle=(0, (5, 1) + (1, 1) * i),
            zorder=2,
        )

    weight_norm = sum(point_to_weight.values())
    for signal_i, color_i in zip(signals, COLORS):
        for point_i, weight_i in point_to_weight.items():
            axis.plot(
                [point_i + signal_i] * 2,
                [ylim[0], numpy.log(weight_i / weight_norm)],
                color=color_i,
            )


    axis.set_xlim(*range_)
    axis.set_ylim(*ylim)
    axis.set_yticks(range(0, -6, -1))
    axis.set_xlabel(r"$\mu$")
    axis.set_ylabel(r"$\log\bar L$ or $\log \pi$")
    axis.xaxis.set_label_coords(0.5, -0.09)
    axis.yaxis.set_label_coords(-0.09, 0.5)
    axis.spines.top.set_visible(False)
    axis.spines.right.set_visible(False)

    axis.tick_params(which="major", direction="in", length=5)
    axis.tick_params(which="minor", direction="in", length=3)

    outpath = "retrograde_models_"
    outpath += "_n" + "_".join(map(str, ns))
    outpath += "_d" + "_".join(map(str, point_to_weight.keys()))
    outpath += "_w" + "_".join(map(str, point_to_weight.values()))
    outpath += ".pdf"
    figure.savefig(outpath)
    print("wrote %r" % outpath)
    pyplot.close(figure)

# utilities


def _logmeanexp(x, axis=0):
    x = numpy.asarray(x)
    norm = numpy.log(x.shape[axis])
    return scipy.special.logsumexp(x, axis=axis) - norm


if __name__ == "__main__":
    main()
