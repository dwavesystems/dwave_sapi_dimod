import sys

import dimod

from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer

__all__ = ['SAPILocalSampler', 'SAPISampler']
__version__ = '0.2'

PY2 = sys.version_info[0] == 2
if PY2:
    # range = xrange
    iteritems = lambda d: d.iteritems()
    # zip = itertools.izip
else:
    iteritems = lambda d: d.items()


class SAPISampler(dimod.TemplateSampler):
    def __init__(self, url, token, solver_name):
        dimod.TemplateSampler.__init__(self)

        self.connection = connection = RemoteConnection(url, token)
        self.solver = solver = connection.get_solver(solver_name)

        self.structure = get_hardware_adjacency(solver)

    @dimod.decorators.ising(1, 2)
    def sample_structured_ising(self, h, J, num_reads=50, **sapi_kwargs):
        """TODO"""
        solver = self.solver

        # sapi expects a list
        h_list = [0.] * solver.properties['num_qubits']
        for v, bias in iteritems(h):
            h_list[v] = bias

        # later we will want to do this asynchronously
        answer = solve_ising(solver, h_list, J, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = answer['solutions']
        energies = answer['energies']
        num_occurrences = answer['num_occurrences']

        samples = ({v: sample[v] for v in h} for sample in solutions)
        sample_data = ({'num_occurrences': n} for n in num_occurrences)

        response = dimod.SpinResponse()
        response.add_samples_from(samples, energies, sample_data)

        return response

    @dimod.decorators.qubo(1)
    def sample_structured_qubo(self, Q, num_reads=50, **sapi_kwargs):
        """TODO"""
        variables = set().union(*Q)

        solver = self.solver

        answer = solve_qubo(solver, Q, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = answer['solutions']
        energies = answer['energies']
        num_occurrences = answer['num_occurrences']

        samples = ({v: sample[v] for v in variables} for sample in solutions)
        sample_data = ({'num_occurrences': n} for n in num_occurrences)

        response = dimod.BinaryResponse()
        response.add_samples_from(samples, energies, sample_data)

        return response

    @dimod.decorators.ising(1, 2)
    @dimod.decorators.ising_index_labels(1, 2)
    def sample_ising(self, h, J, num_reads=50, **sapi_kwargs):
        """TODO"""

        solver = self.solver

        # sapi expects a list, ising_index_labels converted the keys of
        # h to be indices 0, n-1
        h_list = [0.] * len(h)
        for v, bias in iteritems(h):
            h_list[v] = bias

        # find an embedding
        A = self.structure
        S = set(J)
        S.update({(v, v) for v in h})
        embeddings = find_embedding(S, A)

        # now it is possible that h_list might include nodes not in embedding, so let's
        # handle that case here
        if len(h_list) > len(embeddings):
            emb_qubits = set().union(*embeddings)
            while len(h_list) > len(embeddings):
                for v in solver.properties['qubits']:
                    if v not in emb_qubits:
                        embeddings.append([v])
                        emb_qubits.add(v)
                        break

        # embed the problem
        (h0, j0, jc, new_emb) = embed_problem(h_list, J, embeddings, A)

        emb_j = j0.copy()
        emb_j.update(jc)
        answer = solve_ising(solver, h0, emb_j, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = unembed_answer(answer['solutions'], new_emb, 'minimize_energy', h_list, J)
        energies = answer['energies']
        assert len(solutions) == len(energies)
        num_occurrences = answer['num_occurrences']

        samples = ({v: sample[v] for v in h} for sample in solutions)
        sample_data = ({'num_occurrences': n} for n in num_occurrences)

        response = dimod.SpinResponse()
        response.add_samples_from(samples, sample_data=sample_data, h=h, J=J)

        return response


class SAPILocalSampler(SAPISampler):
    def __init__(self, solver_name):
        dimod.TemplateSampler.__init__(self)
        self.solver = solver = local_connection.get_solver(solver_name)
        self.structure = get_hardware_adjacency(solver)
