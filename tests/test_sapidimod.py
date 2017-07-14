import unittest
import logging

from dimod.samplers.tests.generic_sampler_tests import TestSolverAPI
from dwave_sapi_dimod import SAPISampler
from dwave_sapi_dimod import log as dimod_log

# dimod_log.setLevel(logging.DEBUG)


class TestSapiSampler(unittest.TestCase, TestSolverAPI):
    """NOTE: This thing is an exact solver, so we can go ahead and
    check the solution quality
    """
    def setUp(self):
        self.sampler = SAPISampler()

    def test_basic(self):
        h = {0: -1, 1: 0, 2: 0, 3: 0, 4: -1}
        J = {(0, 4): -1}

        response = self.sampler.sample_structured_ising(h, {}, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_structured_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        Q = {(0, 0): -1, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): -1, (0, 4): -2}

        response = self.sampler.sample_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_structured_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

    def test_bug1(self):
        h = {0: -1, 1: 0, 2: 0, 3: 0, 4: -1}
        J = {(0, 4): -1, (1, 4): 1}
        
        response = self.sampler.sample_structured_ising(h, {}, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_structured_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        Q = {(0, 0): -1, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): -1, (0, 4): -2}

        response = self.sampler.sample_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)

        response = self.sampler.sample_structured_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)