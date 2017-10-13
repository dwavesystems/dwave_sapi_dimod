D-Wave SAPI dimod
=================

A dimod wrapper for SAPI.

Installation
------------

For python 2

```
python setup.py install
```

Examples
--------

Loading the module

```python
>>> import dwave_sapi_dimod as sapi
```

Initializing a remote solver

```python
>>> url = "http://myURL"
>>> token = "myToken001"
>>> solver_name = "solver_name"
>>> solver = sapi.SAPISampler(solver_name, url, token)
```

Initializing a local solver.

```python
>>> solver_name = 'c4-sw_optimize'
>>> solver = sapi.SAPILocalSampler(solver_name)
```

sample_qubo and sample_ising work only for problems that fit directly
onto the solver's structure

```python
>>> h = {0: -.1, 4: .1}
>>> J = {(0, 4): -1}
>>> response = solver.sample_ising(h, J)
>>> list(response.samples())
[{0: -1, 4: -1}, {0: 1, 4: 1}, {0: 1, 4: -1}, {0: -1, 4: 1}]
>>> list(response.energies())
[-1.0, -1.0, 0.8, 1.2]
```

For solving arbitrary problems, you need to apply the EmbeddingComposite
layer.

```python
>>> solver = sapi.EmbeddingComposite(sapi.SAPILocalSampler(solver_name))
>>> h = {0: -.1, 1: .1}
>>> J = {(0, 1): -1}
>>> response = solver.sample_ising(h, J)
>>> list(response.samples())
[{0: -1, 1: -1}, {0: 1, 1: 1}, {0: 1, 1: -1}, {0: -1, 1: 1}]
>>> list(response.energies())
[-1.0, -1.0, 0.8, 1.2]
```

See dimod documentation for full description of the response object.

License
-------

See LICENSE.txt
