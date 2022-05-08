"""Define testing utilities."""


def raises(func, exception=BaseException):
    try:
        func()
    except exception:
        return True
    return False
