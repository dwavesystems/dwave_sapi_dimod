import unittest
import logging
import random

from dimod import qubo_energy, ising_energy
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
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, {}, sample), energy)

        response = self.sampler.sample_structured_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        response = self.sampler.sample_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        Q = {(0, 0): -1, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): -1, (0, 4): -2}

        response = self.sampler.sample_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

        response = self.sampler.sample_structured_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

    def test_bug1(self):
        h = {0: -1, 1: 0, 2: 0, 3: 0, 4: -1}
        J = {(0, 4): -1, (1, 4): 1}

        response = self.sampler.sample_structured_ising(h, {}, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, {}, sample), energy)

        response = self.sampler.sample_structured_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        response = self.sampler.sample_ising(h, J, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        Q = {(0, 0): -1, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): -1, (0, 4): -2}

        response = self.sampler.sample_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

        response = self.sampler.sample_structured_qubo(Q, num_reads=10)
        sample = next(response.samples())
        self.assertEqual(sample[0], 1)
        self.assertEqual(sample[4], 1)
        for sample, energy in response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

    def test_bug2(self):
        Q = {(6, 9): 1.125, (11, 11): -2.25, (4, 8): 0.75, (7, 12): 0.75, (2, 8): 1.75,
             (12, 12): -2.625, (0, 10): 1.75, (7, 11): 1.75, (5, 8): 0.75, (6, 7): 1.125,
             (5, 5): -2.25, (3, 10): 0.75, (6, 10): 2.125, (8, 10): 0.75, (9, 11): 0.75,
             (0, 4): 1.75, (3, 5): 2.125, (1, 1): -1.125, (4, 10): 2.125, (2, 6): 0.375,
             (3, 6): 1.75, (5, 11): 2.125, (4, 5): 1.75, (1, 4): 0.375, (0, 12): 1.125,
             (3, 9): 0.75, (2, 3): 0.375, (1, 9): 0.75, (0, 1): 1.75, (3, 12): 0.75,
             (6, 8): 0.75, (1, 12): 0.375, (8, 12): 1.75, (2, 11): 0.375, (9, 9): -2.625,
             (0, 6): 0.75, (0, 9): 2.125, (7, 8): 1.75, (2, 4): 0.75, (9, 12): 2.125,
             (5, 9): 0.375, (4, 7): 0.75, (10, 11): 1.75, (6, 6): -2.625, (5, 6): 2.125,
             (7, 7): -1.875, (8, 9): 1.75, (4, 12): 1.125, (2, 12): 1.75, (3, 7): 2.125,
             (0, 3): 0.375, (1, 2): 1.75, (4, 9): 1.75, (3, 3): -2.25, (2, 9): 2.125,
             (5, 12): 0.75, (4, 4): -2.625, (10, 12): 2.125, (0, 11): 0.375, (7, 10): 0.375,
             (2, 2): -2.25, (1, 10): 0.375, (0, 0): -2.25, (6, 11): 2.125, (8, 11): 0.75,
             (2, 10): 1.125, (9, 10): 2.125, (0, 5): 0.375, (6, 12): 1.75, (0, 8): 0.75,
             (11, 12): 1.75, (4, 11): 1.125, (7, 9): 0.375, (2, 7): 0.375, (5, 10): 1.125,
             (4, 6): 2.125, (10, 10): -2.625, (3, 4): 1.75, (5, 7): 2.125, (3, 8): 1.75,
             (3, 11): 1.125, (1, 8): 0.375, (8, 8): -2.25, (0, 2): 2.125}

        nodes = set().union(*Q)

        response = self.sampler.sample_qubo(Q)

        for sample, energy in response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)
            print sample, energy

    def test_bug3(self):
        """disconnected graph"""
        h = {v: random.uniform(-2, 2) for v in [0, 1, 2, 3, 4, 5, 6, 7]}

        J = {edge: random.uniform(-1, 1) for edge in [(1, 3), (1, 5), (1, 7), (2, 3), (2, 4),
                                                      (2, 6), (3, 5), (4, 5), (5, 6)]}

        response = self.sampler.sample_ising(h, J)
        print response