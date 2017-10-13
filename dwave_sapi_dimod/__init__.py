from __future__ import absolute_import

import sys

__version__ = '0.2'

_PY2 = sys.version_info[0] == 2

from dwave_sapi_dimod.samplers import *
from dwave_sapi_dimod.composites import *
