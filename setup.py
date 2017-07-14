from setuptools import setup, find_packages

from dwave_sapi_dimod import __version__

setup(
    name='dwave_sapi_dimod',
    version=__version__,
    packages=find_packages(),
    install_requires=['dimod']
)
