"""
dwave_sapi_dimod
================

A dimod wrapper for SAPI.

Examples:

Initializing a remote solver

>>> url = "http://myURL"
>>> token = "myToken001"
>>> solver_name = "solver_name"
>>> solver = SAPISampler(solver_name, url, token)]

Initializing a local solver.

>>> solver_name = 'c4-sw_optimize'
>>> solver = SAPILocalSolver(solver_name)

sample_qubo and sample_ising work for arbitrarily structured Ising
problems and QUBOs.

>>> h = {0: -.1, 1: .1}
>>> J = {(0, 1): -1}
>>> response = solver.solve_ising(h, J)
>>> list(response.samples())
[{0: -1, 1: -1}, {0: 1, 1: -1}]
>>> list(response.energies())
[-1, -1]

sample_structured_ising and sample_structured_qubo works only for
problems that fit directly onto the solver's structure

>>> h = {0: -.1, 4: .1}
>>> J = {(0, 4): -1}
>>> response = solver.solve_ising(h, J)
>>> list(response.samples())
[{0: -1, 4: -1}, {0: 1, 4: -1}]
>>> list(response.energies())
[-1, -1]

See dimod documentation for full description of the response object.

"""

import sys

import dimod

from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer

__all__ = ['SAPILocalSampler', 'SAPISampler']


if _PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


class SAPISampler(dimod.TemplateSampler):
    """dimod wrapper for a SAPI remote solver.

    Args:
        solver_name (str): The string name of the desired solver, as
            returned by `solver_names`.
        url (str): SAPI URL.
        token (str): API token.
        proxy_url (str): Proxy URL.

    Attributes:
        structure (set): The set of edges available to the solver.

    Notes:
        See QUBIST documentation at https://dw2x.dwavesys.com/ for
        further details.

    """
    def __init__(self, solver_name, url, token, proxy_url=None):
        dimod.TemplateSampler.__init__(self)

        if proxy_url is None:
            self.connection = connection = RemoteConnection(url, token)
        else:
            self.connection = connection = RemoteConnection(url, token, proxy_url)

        self.solver = solver = connection.get_solver(solver_name)

        self.structure = get_hardware_adjacency(solver)

    @dimod.decorators.qubo(1)
    def sample_structured_qubo(self, Q, num_reads=50, **sapi_kwargs):
        """Solve the QUBO.

        Args:
            Q (dict): A dictionary defining the QUBO. Should be of the
                form {(u, v): bias} where u, v are variables and bias
                is numeric. The edges in Q must be a subset of those
                given in the `structure` parameter.

            Additional keyword parameters are the same as for
            SAPI's solve_qubo function, see QUBIST documentation.

        Returns:
            :obj:`BinaryResponse`

        Notes:
            See QUBIST documentation at https://dw2x.dwavesys.com/ for
            further details.

        """
        variables = set().union(*Q)

        solver = self.solver

        answer = solve_qubo(solver, Q, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = answer['solutions']
        energies = answer['energies']

        # convert the answer to a dict
        samples = ({v: sample[v] for v in variables} for sample in solutions)

        # if information about the number of occurrences is returned, include it
        if 'num_occurrences' in answer:
            num_occurrences = answer['num_occurrences']
            sample_data = ({'num_occurrences': n} for n in num_occurrences)
        else:
            sample_data = ({} for __ in solutions)

        response = dimod.BinaryResponse()
        response.add_samples_from(samples, energies, sample_data)

        return response

    @dimod.decorators.ising(1, 2)
    @dimod.decorators.ising_index_labels(1, 2)
    def sample_ising(self, h, J, num_reads=50, **sapi_kwargs):
        """Solve ths Ising problem.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is a
                variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias, ...}
                where u, v are variables in the Ising problem and
                bias is the quadratic bias associated with u, v.

            Additional keyword parameters are the same as for
            SAPI's solve_ising function, see QUBIST documentation.

        Returns:
            :obj:`SpinResponse`

        Notes:
            See QUBIST documentation at https://dw2x.dwavesys.com/ for
            further details.

        """

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

        if J and not embeddings:
            raise Exception('No embedding found')

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
        h0, j0, jc, new_emb = embed_problem(h_list, J, embeddings, A)

        emb_j = j0.copy()
        emb_j.update(jc)
        answer = solve_ising(solver, h0, emb_j, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = unembed_answer(answer['solutions'], new_emb, 'minimize_energy', h_list, J)
        energies = answer['energies']
        assert len(solutions) == len(energies)

        samples = ({v: sample[v] for v in h} for sample in solutions)

        if 'num_occurrences' in answer:
            num_occurrences = answer['num_occurrences']
            sample_data = ({'num_occurrences': n} for n in num_occurrences)
        else:
            sample_data = ({} for __ in range(len(solutions)))

        response = dimod.SpinResponse()
        response.add_samples_from(samples, sample_data=sample_data, h=h, J=J)

        return response


class SAPILocalSampler(SAPISampler):
    """dimod wrapper for a SAPI local solver.

    Args:
        solver_name (str): The string name of the desired solver, as
            returned by `solver_names`.

    Attributes:
        structure (set): The set of edges available to the solver.

    Notes:
        See QUBIST documentation at https://dw2x.dwavesys.com/ for
        further details.

    """
    def __init__(self, solver_name):
        dimod.TemplateSampler.__init__(self)
        self.solver = solver = local_connection.get_solver(solver_name)
        self.structure = get_hardware_adjacency(solver)
