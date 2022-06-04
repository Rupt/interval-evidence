"""Define testing utilities."""
import sys
import traceback


def raises(func, exception=BaseException):
    """Return true iff func raises exception."""
    try:
        func()
    except exception:
        return True
    return False


def test_funcs(funcs):
    """Print information from calls to functions in funcs."""
    for func in funcs:
        try:
            func()
            print(end=".", flush=True)
        except AssertionError:
            print(end="!", flush=True)
            print(file=sys.stderr)
            traceback.print_exc()

    print()
