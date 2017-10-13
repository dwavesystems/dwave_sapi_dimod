import dimod

from dwave_sapi2.remote import RemoteConnection
from dwave_sapi2.local import local_connection
from dwave_sapi2.core import solve_ising, solve_qubo
from dwave_sapi2.util import get_hardware_adjacency
from dwave_sapi2.embedding import find_embedding, embed_problem, unembed_answer

from dwave_sapi_dimod import _PY2

if _PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


class EmbeddingComposite(dimod.TemplateComposite):
    def __init__(self, sampler):
        # puts sampler into self.children
        dimod.TemplateComposite.__init__(self, sampler)

        self._child = sampler  # faster access than self.children[0]

        # structure becomes None
        self.structure = None

    @dimod.decorators.ising(1, 2)
    @dimod.decorators.ising_index_labels(1, 2)
    def sample_ising(self, h, J, **sapi_kwargs):

        sampler = self._child

        # sapi expects a list, ising_index_labels converted the keys of
        # h to be indices 0, n-1
        h_list = [0.] * len(h)
        for v, bias in iteritems(h):
            h_list[v] = bias

        # find an embedding
        (__, A) = sampler.structure
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

        if 'chains' in sampler.solver.properties['parameters'] and 'chains' not in sapi_kwargs:
            sapi_kwargs['chains'] = embeddings

        emb_response = sampler.sample_ising(h0, emb_j, **sapi_kwargs)

        answers = [[sample[i] for i in range(len(sample))] for sample in emb_response]

        # parse the answers
        solutions = unembed_answer(answers, new_emb, 'minimize_energy', h_list, J)

        samples = ({v: sample[v] for v in h} for sample in solutions)

        sample_data = (data for __, data in emb_response.samples(data=True))

        response = dimod.SpinResponse()
        response.add_samples_from(samples, sample_data=sample_data, h=h, J=J)

        return response
