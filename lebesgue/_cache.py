"""Cache function calls."""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class MutableCache:
    """Wrap a function with only positional arguments with a dict of results
    which can be used to cache or override return values.
    """

    func: Callable
    cache: dict = field(default_factory=dict)

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

    def put(self, *args):
        """Return a function which sets the return value for args."""

        def put_args(result):
            self.cache[args] = result
            return result

        return put_args
