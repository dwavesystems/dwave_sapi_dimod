from setuptools import setup, find_packages

from dwave_sapi_dimod import __version__

setup(
    name='dwave_sapi_dimod',
    version=__version__,
    py_modules=['dwave_sapi_dimod'],
    install_requires=['dimod>=0.3.0']
)
