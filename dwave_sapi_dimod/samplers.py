"""


"""
import random

import dimod

from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer

from dwave_sapi_dimod import _PY2

__all__ = ['SAPILocalSampler', 'SAPISampler']


if _PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


class SAPILocalSampler(dimod.TemplateSampler):
    """dimod wrapper for a SAPI local solver.

    Args:
        solver_name (str): The string name of the desired solver, as
            returned by `solver_names`.

    Attributes:
        structure (tuple): (nodes, edges), the set of nodes and edges
            available to the solver.

    Notes:
        See QUBIST documentation at https://dw2x.dwavesys.com/ for
        further details.

    """
    def __init__(self, solver_name):
        dimod.TemplateSampler.__init__(self)
        self.solver = solver = local_connection.get_solver(solver_name)
        edges = get_hardware_adjacency(solver)
        self.structure = (set().union(*edges), edges)

    @dimod.decorators.qubo(1)
    def sample_qubo(self, Q, num_reads=50, **sapi_kwargs):
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

        if not all(isinstance(v, int) for v in variables):
            raise ValueError('all variables must be index labeled')

        solver = self.solver

        # for whatever reason sapi needs Q to be cleaned of empty values
        Q = {edge: bias for edge, bias in iteritems(Q) if bias != 0.0}

        answer = solve_qubo(solver, Q, num_reads=num_reads, **sapi_kwargs)

        # parse the answers
        solutions = answer['solutions']
        energies = answer['energies']

        # convert the answer to a dict. sapi returnes answers that were 'off' as 3,
        # so let's just choose a random value for them
        samples = ({v: sample[v] if sample[v] != 3 else random.choice((0, 1))
                    for v in variables} for sample in solutions)

        # if information about the number of occurrences is returned, include it
        if 'num_occurrences' in answer:
            num_occurrences = answer['num_occurrences']
            sample_data = ({'num_occurrences': n} for n in num_occurrences)
        else:
            sample_data = ({} for __ in solutions)

        response = dimod.BinaryResponse()
        response.add_samples_from(samples, energies, sample_data)

        return response


class SAPISampler(SAPILocalSampler):
    """dimod wrapper for a SAPI remote solver.

    Args:
        solver_name (str): The string name of the desired solver, as
            returned by `solver_names`.
        url (str): SAPI url.
        token (str): API token.
        proxy_url (str): Proxy url.

    Attributes:
        structure (tuple): (nodes, edges), the set of nodes and edges
            available to the solver.

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

        edges = get_hardware_adjacency(solver)
        self.structure = (set().union(*edges), edges)