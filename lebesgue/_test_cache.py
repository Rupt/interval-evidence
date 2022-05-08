from ._cache import MutableCache


def test_caching():
    def func(x, y):
        return x + y

    cache = MutableCache(func)

    assert func(1, 2) == cache(1, 2)
    assert func(3.0, 4.0) == cache(3.0, 4.0)

    assert cache((1,), (2, 3)) is cache((1,), (2, 3))

    # could pass all above by always returning the same value
    assert cache(3.0, 4.0) != cache(1, 2)


def test_put():
    def func(x):
        return 1

    cache = MutableCache(func)
    cache.put(0)(-1)

    assert cache(1) == func(1)
    assert cache(2) == func(2)
    assert cache(0) == -1

    @cache.put(3)
    def three(x):
        pass

    assert cache(3) is three
