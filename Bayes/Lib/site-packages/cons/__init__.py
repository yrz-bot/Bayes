from importlib.metadata import version

from . import unify  # noqa: F401
from .core import car, cdr, cons  # noqa: F401

__version__ = version("cons")
