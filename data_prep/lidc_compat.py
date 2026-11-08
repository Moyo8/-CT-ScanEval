"""Import pylidc on modern NumPy / Python.

pylidc 0.2.3 (2019) still uses the NumPy aliases np.int / np.bool / np.float
that were removed in NumPy 1.24. Restoring them before import is enough.
It also uses configparser.SafeConfigParser (removed in Python 3.12).
Only the data-preparation scripts need this; graders and the harness do not.
"""
import configparser
import warnings

import numpy as np

# Python 3.12 removed configparser.SafeConfigParser, which pylidc still calls.
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

for _name, _type in (("int", int), ("bool", bool), ("float", float)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pylidc as pl  # noqa: E402
from pylidc.utils import consensus  # noqa: E402

__all__ = ["pl", "consensus"]
