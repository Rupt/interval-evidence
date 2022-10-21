"""Define standard plotting content."""
from matplotlib import pyplot

INCHES_PER_CM = 2.54

# extracted  with:
# \usepackage{layouts}
# % ...
# \printinunitsof{cm}\prntlen{\textwidth}
# \printinunitsof{cm}\prntlen{\columnwidth}
# % https://tex.stackexchange.com/a/39385
WIDTH_TEXT_CM = 15.99773
WIDTH_COLUMN_CM = 7.5989

WIDTH_TEXT = WIDTH_TEXT_CM / INCHES_PER_CM
WIDTH_COLUMN = WIDTH_COLUMN_CM / INCHES_PER_CM


def default_init():
    pyplot.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
        }
    )
