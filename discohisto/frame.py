"""Work with namespaces of arrays as minimal dataframes.

A frame is a SimpleNamespace of numpy arrays.

e.g.


frame = SimpleNamespace(
    foo=numpy.array([1, 2, 3]),
    bar=numpy.array([3.1, 3.2, 3.3]),
)

dump(frame, "test.csv")

print(load("test.csv"))

"""
from types import SimpleNamespace

import numpy

DELIMITER = ","


def dump(frame, path):
    """Serialize frame to csv(.gz) at path."""
    frame_dict = frame.__dict__

    keys = DELIMITER.join(frame_dict.keys())
    dtypes = DELIMITER.join([array.dtype.str for array in frame_dict.values()])

    return numpy.savetxt(
        path,
        list(frame_dict.values()),
        delimiter=DELIMITER,
        header=keys + "\n" + dtypes,
    )


def load(path):
    """Load a frame from path."""
    with open(path) as file_:
        comment_keys = file_.readline()
        comment_dtypes = file_.readline()
        arrays = numpy.loadtxt(file_, delimiter=DELIMITER)

    # comments begin "# " and end in "\n"
    keys = comment_keys[2:-1].split(DELIMITER)
    dtypes = comment_dtypes[2:-1].split(DELIMITER)

    frame_dict = {
        key: array.astype(dtype)
        for key, dtype, array in zip(keys, dtypes, arrays)
    }

    return SimpleNamespace(**frame_dict)
