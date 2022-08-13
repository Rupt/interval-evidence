"""Work with namespaces of arrays as minimal dataframes.

A frame is a SimpleNamespace of numpy arrays of strings, integers or floats.

"""
import ast
from types import SimpleNamespace

import numpy


def dump(frame, path):
    """Serialize frame to csv(.gz) at path."""
    frame_dict = frame.__dict__

    # tuple repr surrounds it in brackets
    keys = repr(tuple(frame_dict.keys()))[1:-1]

    dtypes_iter = (array.dtype.str for array in frame_dict.values())
    dtypes = repr(tuple(dtypes_iter))[1:-1]

    with open(path, "w") as file_:
        file_.write(f"# {keys}\n")
        file_.write(f"# {dtypes}\n")

        for items in zip(*frame_dict.values()):
            row = repr(items)[1:-1]
            file_.write(f"{row}\n")


def load(path):
    """Load a frame from path."""
    with open(path) as file_:
        # comments begin "# " and end "\n"
        comment_keys = file_.readline()[2:-1]
        comment_dtypes = file_.readline()[2:-1]
        keys = ast.literal_eval(f"({comment_keys})")
        dtypes = ast.literal_eval(f"({comment_dtypes})")
        assert len(keys) == len(dtypes)

        columns = [[] for _ in keys]

        for row in file_:
            items = ast.literal_eval(f"({row})")
            for column, item in zip(columns, items):
                column.append(item)

    frame_dict = {
        key: numpy.array(column, dtype=dtype)
        for key, dtype, column in zip(keys, dtypes, columns)
    }

    return SimpleNamespace(**frame_dict)


def _test():
    frame = SimpleNamespace(
        test=numpy.array(["a", "b", "c"]),
        foo=numpy.array([1, 2, 3]),
        bar=numpy.array([3.1, 3.2, 3.3]),
    )

    dump(frame, "test.csv")

    print(load("test.csv"))


if __name__ == "__main__":
    _test()
