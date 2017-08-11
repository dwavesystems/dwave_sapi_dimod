import unittest
import random

import dimod

from dwave_sapi_dimod import SAPISampler, SAPILocalSampler

# ##################################
# this should be depreciated
import sapi_token
url, token = sapi_token.token()
solver = 'c4-sw_optimize'
# ##################################


class MethodTests:
    def test_simple_problem(self):
        # solver is an exact solver so can check results directly
        sampler = self.sampler

        h = {0: 1}
        J = {(0, 4): -1}

        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], -1)
        self.assertEqual(min_sample[4], -1)

        Q = {(0, 0): 1, (0, 4): -1.2, (4, 4): .1}

        response = sampler.sample_structured_qubo(Q)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], 1)
        self.assertEqual(min_sample[4], 1)

        # now an unembded
        h = {0: 1}
        J = {(0, 1): -1}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], -1)
        self.assertEqual(min_sample[1], -1)

        Q = {(0, 0): 1, (0, 1): -1.2, (1, 1): .1}

        response = sampler.sample_qubo(Q)

        # check that the min energy sample has the answer we expected
        min_sample = next(iter(response))
        self.assertEqual(min_sample[0], 1)
        self.assertEqual(min_sample[1], 1)

    def test_small_edge_cases(self):
        sampler = self.sampler

        # empty

        h = {}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)
        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)
        response = sampler.sample_structured_qubo(Q)
        self.check_binary_response(response, Q)

        # one node

        h = {0: 1}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)
        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 0): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)
        response = sampler.sample_structured_qubo(Q)
        self.check_binary_response(response, Q)

        # one edge

        h = {}
        J = {(0, 4): -1}  # must be a real coupler to test structured
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)
        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 4): 1}   # must be a real coupler to test structured
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)
        response = sampler.sample_structured_qubo(Q)
        self.check_binary_response(response, Q)

        # two nodes

        h = {0: -1, 6: 1}
        J = {}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)
        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 0): 1, (6, 6): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)
        response = sampler.sample_structured_qubo(Q)
        self.check_binary_response(response, Q)

    def test_triangle(self):
        # smallest graph that will require a real embedding
        sampler = self.sampler

        h = {}
        J = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

    def test_random_problem(self):
        sampler = self.sampler

        edges = sampler.structure

        nodes = set().union(*edges)

        h = {v: random.uniform(-2, 2) for v in nodes}
        J = {(u, v): random.uniform(-1, 1) for u, v in edges if u < v}

        response = sampler.sample_structured_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = J.copy()
        Q.update({(v, v): h[v] for v in h})
        response = sampler.sample_structured_qubo(Q)
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


class TestSAPISampler(unittest.TestCase, MethodTests):
    def test_connect(self):
        """ways of initializing the sampler"""
        sampler = SAPISampler(url, token, solver)
        # TODO, check defaults

    def setUp(self):
        self.sampler = SAPISampler(url, token, solver)


class TestSAPILocalSampler(unittest.TestCase, MethodTests):
    def setUp(self):
        self.sampler = SAPILocalSampler(solver)
