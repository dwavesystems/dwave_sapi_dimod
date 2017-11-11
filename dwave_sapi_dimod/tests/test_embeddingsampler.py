"""
Tests for EmbeddingComposite(SAPISampler()) and EmbeddingComposite(SAPILocalSampler()).

Both of these samplers are unstructured.
"""

import unittest
import random
import itertools

import dimod

import dwave_sapi_dimod as sapi

try:
    from sapi_token import url, token, solver_name
    _sapitoken = True
except ImportError:
    _sapitoken = False


class TestEmbeddingCompositeSAPILocalSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = sapi.EmbeddingComposite(sapi.SAPILocalSampler('c4-sw_optimize'))

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

    def test_basic(self):
        sampler = self.sampler

        # now an unembeded
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

    def test_disconnected(self):
        sampler = self.sampler

        h = {5: -1}
        J = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
        response = sampler.sample_ising(h, J)
        self.check_spin_response(response, h, J)

        Q = {(0, 1): 1, (1, 2): 1, (0, 2): 1, (5, 5): 1}
        response = sampler.sample_qubo(Q)
        self.check_binary_response(response, Q)

    def check_spin_response(self, response, h, J):
        variables = set(h)
        variables.update(set().union(*J))

        for sample, energy in response.items():
            for v in variables:
                self.assertIn(v, sample)
            self.assertLessEqual(abs(dimod.ising_energy(h, J, sample) - energy), 10**-5)

            self.assertEqual(len(sample), len(variables))

    def check_binary_response(self, response, Q):
        variables = set().union(*Q)

        for sample, energy in response.items():
            for v in variables:
                self.assertIn(v, sample)
            self.assertLessEqual(abs(dimod.qubo_energy(Q, sample) - energy), 10**-5)

            self.assertEqual(len(sample), len(variables))

    def test_embedding_tag(self):
        # check reusing the same embedding tag

        sampler = self.sampler

        # get an embedding on a complete K10
        h = {v: .01 * v for v in range(-5, 5)}
        J = {(u, v): 1 for u, v in itertools.combinations(h, 2)}

        responses = []
        for __ in range(10):
            responses.append(sampler.sample_ising(h, J, embedding_tag='K10'))




@unittest.skipUnless(_sapitoken, "need a sapi token for testing")
class TestEmbeddingCompositeSAPISampler(TestEmbeddingCompositeSAPILocalSampler):
    def setUp(self):
        self.sampler = sapi.EmbeddingComposite(sapi.SAPISampler(solver_name, url, token))
