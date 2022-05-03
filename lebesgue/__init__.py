# importing _cache prepares some compiled code; discard its reference
from . import _cache, likelihood, prior
from ._bayes import Model

del _cache
