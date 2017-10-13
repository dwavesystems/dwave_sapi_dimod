"""
Tests for SAPISampler and SAPILocalSampler.

Both of these samplers are structured, that is they have a problem
graph.
"""

import unittest
import random

import dimod

from dwave_sapi_dimod import SAPISampler, SAPILocalSampler

try:
    from sapi_token import url, token, solver_name
    _sapitoken = True
except ImportError:
    _sapitoken = False


class TestSAPILocalSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = SAPILocalSampler('c4-sw_optimize')

    def test_small_problem(self):
        # solver is an exact solver so can check results directly
        sampler = self.sampler

        h = {0: 1}
        J = {(0, 4): -1}

        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], -1)
        self.assertEqual(min_sample[4], -1)

        Q = {(0, 0): 1, (0, 4): -1.2, (4, 4): .1}

        response = sampler.sample_qubo(Q)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], 1)
        self.assertEqual(min_sample[4], 1)

    def test_trivial_problem(self):
        sampler = self.sampler

        # empty

        h = {}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

        # one node

        h = {0: 1}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 0): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

        # one edge

        h = {}
        J = {(0, 4): -1}  # must be a real coupler to test structured
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 4): 1}   # must be a real coupler to test structured
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

        # two nodes

        h = {0: -1, 6: 1}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 0): 1, (6, 6): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

    def test_random_problem(self):
        sampler = self.sampler

        nodes, edges = sampler.structure

        h = {v: random.uniform(-2., 2.) for v in nodes}
        J = {(u, v): random.uniform(-1., 1.) for u, v in edges if u < v}

        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = J.copy()
        Q.update({(v, v): h[v] for v in h})
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

    def check_spin_response(self, response, h, J):
        variables = set(h)
        variables.update(set().union(*J))

        for sample, energy in response.items():
            for v in variables:
                self.assertIn(v, sample)
            self.assertLessEqual(abs(dimod.ising_energy(h, J, sample) - energy), 10**-5)

    def check_binary_response(self, response, Q):
        variables = set().union(*Q)

        for sample, energy in response.items():
            for v in variables:
                self.assertIn(v, sample)
            self.assertLessEqual(abs(dimod.qubo_energy(Q, sample) - energy), 10**-5)


@unittest.skipUnless(_sapitoken, "need a sapi token for testing")
class TestSAPISampler(TestSAPILocalSampler):
    def setUp(self):
        self.sampler = SAPISampler(solver_name, url, token)
