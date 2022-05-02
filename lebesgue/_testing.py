"""Define testing utilities."""


def raises(func, exception=Exception):
    try:
        func()
    except exception:
        return True
    return False
