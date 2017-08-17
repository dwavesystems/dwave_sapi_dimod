dwave_sapi_dimod
================

A dimod wrapper for SAPI.

Installation
------------

For python 2

```
python setup.py install
```

Examples
--------

Initializing a remote solver

```python
>>> url = "http://myURL"
>>> token = "myToken001"
>>> solver_name = "solver_name"
>>> solver = SAPISampler(solver_name, url, token)]
```

Initializing a local solver.

```python
>>> solver_name = 'c4-sw_optimize'
>>> solver = SAPILocalSolver(solver_name)
```

sample_qubo and sample_ising work for arbitrarily structured Ising
problems and QUBOs.

```python
>>> h = {0: -.1, 1: .1}
>>> J = {(0, 1): -1}
>>> response = solver.solve_ising(h, J)
>>> list(response.samples())
[{0: -1, 1: -1}, {0: 1, 1: -1}]
>>> list(response.energies())
[-1, -1]
```

sample_structured_ising and sample_structured_qubo works only for
problems that fit directly onto the solver's structure

```python
>>> h = {0: -.1, 4: .1}
>>> J = {(0, 4): -1}
>>> response = solver.solve_ising(h, J)
>>> list(response.samples())
[{0: -1, 4: -1}, {0: 1, 4: -1}]
>>> list(response.energies())
[-1, -1]
```

See dimod documentation for full description of the response object.

License
-------

See LICENSE.txt
