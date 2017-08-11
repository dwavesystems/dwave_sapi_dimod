import unittest

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
