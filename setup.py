from setuptools import setup, find_packages

from dwave_sapi_dimod import __version__


packages = ['dwave_sapi_dimod']

setup(
    name='dwave_sapi_dimod',
    version=__version__,
    packages=packages,
    install_requires=['dimod>=0.3.0',
                      'dwave_sapi2'],
    license='Apache 2.0',
)
