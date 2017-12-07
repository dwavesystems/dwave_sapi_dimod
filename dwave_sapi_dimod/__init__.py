from __future__ import absolute_import

import sys as _sys

__version__ = '0.3.1'

_PY2 = _sys.version_info[0] == 2

from dwave_sapi_dimod.samplers import *
import dwave_sapi_dimod.samplers

from dwave_sapi_dimod.composites import *
import dwave_sapi_dimod.composites
