import unittest

from dimod.samplers.tests.generic_sampler_tests import TestSolverAPI

from dwave_sapi_dimod import SAPISampler

class TestSapiSampler(unittest.TestCase, TestSolverAPI):
    def setUp(self):
        self.sampler = SAPISampler()


    def test_basic(self):
        h = {0: -1, 1: 0, 2: 0, 3: 0, 4: -1}
        J = {(0, 4): -1}

        response = self.sampler.sample_structured_ising(h, {}, num_reads=10)
        
        response = self.sampler.sample_structured_ising(h, J, num_reads=10)

        response = self.sampler.sample_ising(h, J, num_reads=10)

        # print response

