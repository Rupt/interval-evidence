"""Dump plots comparing our limits to those reported.

Usage:

python report/plot_limits.py

"""
import matplotlib
from matplotlib import pyplot

from report import frame, plot_lib


def main():
    frame_ = frame.load("report/results.csv")

    plot_lib.default_init()
    plot_limits(frame_, "cabinetry", "Cabinetry truncated normal")
    plot_limits(frame_, "normal", "Autodiff truncated normal")
    plot_limits(frame_, "normal_log", "Autodiff log normal")
    plot_limits(frame_, "linspace", "Profile")
    plot_limits(frame_, "delta", "Best fit delta")
    plot_limits(frame_, "mcmc", "MCMC histogram")


def plot_limits(frame_, label, description_prior, lim=(1.5, 350)):
    reported_obs = frame_.reported_s95obs
    label_2obs = getattr(frame_, "limit_%s_2obs" % label)
    label_3obs = getattr(frame_, "limit_%s_3obs" % label)
    assert lim[0] < min(min(reported_obs), min(label_2obs), min(label_3obs))
    assert lim[1] > max(max(reported_obs), max(label_2obs), max(label_3obs))

    figure, axis = pyplot.subplots(
        figsize=(plot_lib.WIDTH_COLUMN, plot_lib.WIDTH_COLUMN * 0.98),
        dpi=400,
        gridspec_kw={
            "top": 0.999,
            "right": 0.999,
            "bottom": 0.11,
            "left": 0.12,
        },
    )

    (scatter2,) = axis.plot(
        reported_obs,
        label_2obs,
        color="r",
        lw=0,
        marker="o",
        markersize=0.8,
    )
    (scatter3,) = axis.plot(
        reported_obs,
        label_3obs,
        color="b",
        lw=0,
        marker="o",
        markersize=0.8,
    )
    axis.plot(lim, lim, "k", lw=0.5, zorder=0.5, markersize=0)

    axis.text(
        0.03,
        0.99,
        f"Prior: {description_prior}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
    )

    axis.legend(
        [_marker_from(scatter3, 3), _marker_from(scatter2, 3)],
        [r"$T^3_\mathrm{obs}$", r"$T^2_\mathrm{obs}$"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.92),
        numpoints=3,
        framealpha=0,
        handletextpad=0.4,
        borderpad=0,
        borderaxespad=0,
    )

    axis.set_aspect("equal")
    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_xlim(*lim)
    axis.set_ylim(*lim)
    axis.set_xlabel(r"$S^{95}_\mathrm{obs}$")
    axis.set_ylabel(r"$T^*_\mathrm{obs}$")
    axis.xaxis.set_label_coords(0.5, -0.06)
    axis.yaxis.set_label_coords(-0.09, 0.5)
    axis.spines.top.set_visible(False)
    axis.spines.right.set_visible(False)

    # https://stackoverflow.com/a/51213884
    # default tick sizes {major: 3.5, minor: 2.0}
    # axis.xaxis.majorTicks[0].tick1line.get_markersize())
    # axis.xaxis.minorTicks[0].tick1line.get_markersize())
    axis.tick_params(which="major", direction="in", length=5)
    axis.tick_params(which="minor", direction="in", length=3)

    outpath = f"limits_{label}.pdf"
    figure.savefig(outpath)
    print("wrote %r" % outpath)
    pyplot.close(figure)


def _marker_from(line, size):
    marker = matplotlib.lines.Line2D([], [])
    marker.update_from(line)
    marker.set_markersize(size)
    return marker


if __name__ == "__main__":
    main()
