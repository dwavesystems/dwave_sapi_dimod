import sys
import random
import itertools

from dimod import DiscreteModelSampler, SpinResponse, ising_energy, qubo_to_ising
from dimod.decorators import ising, ising_index_labels, qubo, qubo_index_labels

from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_chimera_adjacency, qubo_to_ising
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer

__all__ = ['SAPISampler']

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange
    iteritems = lambda d: d.iteritems()
    zip = itertools.izip
else:
    iteritems = lambda d: d.items()

solver = local_connection.get_solver("c4-sw_optimize")
A = get_chimera_adjacency(4, 4, 4)

class SAPISampler(DiscreteModelSampler):

    @ising(1, 2)
    @ising_index_labels(1, 2)
    def sample_ising(self, h, J, num_reads=1, **sapi_kwargs):

        response = SpinResponse()

        # if J is empty, then we are already done
        if not J:
            for __ in range(num_reads):
                response.add_sample({v: _linear_to_spin(bias) for v, bias in iteritems(h)}, h=h, J=J)
            return response

        # add number of reads to SAPI
        sapi_kwargs['num_reads'] = num_reads
        
        # convert h to a list
        h_list = [h[i] for i in range(len(h))]

        # we need to know what variables we need
        S = {(v, v) for v in h}
        S.add(*J)

        # find an embedding
        embeddings = find_embedding(S, A)
        [h0, j0, jc, embeddings] = embed_problem(h_list, J, embeddings, A)

        # solve on sapi
        j_emb = j0
        j_emb.update(jc)
        answer = solve_ising(solver, h0, j_emb, **sapi_kwargs)

        # unembed
        result = unembed_answer(answer['solutions'], embeddings, broken_chains='minimize_energy', h=h_list, j=J)

        # parse the response
        for soln, energy in zip(answer['solutions'], answer['energies']):
            sample = {v: soln[v] for v in h}
            response.add_sample(sample, energy)

        return response

    response = SpinResponse()
    @ising(1, 2)
    @ising_index_labels(1, 2)
    def sample_structured_ising(self, h, J, num_reads=1, **sapi_kwargs):

        response = SpinResponse()

        # if J is empty, then we are already done
        if not J:
            for __ in range(num_reads):
                response.add_sample({v: _linear_to_spin(bias) for v, bias in iteritems(h)}, h=h, J=J)
            return response

        # add number of reads to SAPI
        sapi_kwargs['num_reads'] = num_reads

        # convert h to a list
        h_list = [h[i] for i in range(len(h))]

        # get the answer from sapi solver
        answer = solve_ising(solver, h_list, J, **sapi_kwargs)

        # parse the response
        for soln, energy in zip(answer['solutions'], answer['energies']):
            sample = {v: soln[v] for v in h}
            response.add_sample(sample, energy)

        return response

    @qubo(1)
    def sample_qubo(self, Q, **args):
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.sample_ising(h, J, **args)
        return spin_response.as_binary(offset)

    def sample_structured_qubo(self, Q, **args):
        return self.sample_qubo(Q, **args)


def _linear_to_spin(bias):
    if bias < 0:
        return 1
    elif bias > 0:
        return -1
    else:
        return random.choice((-1, 1))