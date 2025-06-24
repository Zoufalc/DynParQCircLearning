# Dynamic parameterized quantum circuits: expressive and barren-plateau free

This repository provides the code for reproducing the results in [arXiv:2411.05760](https://arxiv.org/abs/2411.05760).

### Description
The paper discusses the properties of a class of dynamic parametrized quantum circuits (DPQC) as an ansatz---parameterized circuits containing intermediate measurements and feedforward operations---for variational quantum algorithms.
Specifically, the investigation focuses on their classical simulability and the hardness of optimization.
In particular, it is shown that these architectures are promising candidates for a variety of applications because they:
1. Provably do not suffer from barren plateaus. 
2. Are expressive enough to describe arbitrarily deep unitary quantum circuits. 
3. Are competitive with state of the art methods for the preparation of ground states and facilitate the representation of nontrivial thermal states.

### Structure

- `src`
  - `examples`: Files that illustrate the code execution.
  - `utilities`: Files that hold the utility functionality needed to execute the code.
  - `find_gibbs_fidelity`: Functionality to approximate a Gibbs state using DPQCs as ansatz based on the infidelity as cost function.
  - `find_gibbs_varqite`: Functionality to approximate a Gibbs state using DPQCs as ansatz based on variational quantum imaginary time evolution.
  - `find_gs`: Functionality to approximate a ground state using DPQCs as ansatz with a sparse or local observable as a cost function.

### Contributors and acknowledgement

Abhinav Deshpande <abhinav.deshpande@ibm.com>
<br/>
Marcel Hinsche
<br/>
Sona Najafi
<br/>
Kunal Sharma
<br/>
Ryan Sweke
<br/>
Christa Zoufal <ouf@zurich.ibm.com>