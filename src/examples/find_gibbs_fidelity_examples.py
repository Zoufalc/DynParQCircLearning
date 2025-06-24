# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Gibbs state preparation based on fidelity maximization using the dynamic parameterized quantum ansatz
for a transverse field Ising model."""

import numpy as np
import csv
import os


np.set_printoptions(legacy='1.21')
from joblib import Parallel, delayed

from src.utilities.generate_ising_hamiltonian import tfi_hamiltonian
from src.utilities.generate_ansatz import *
from src.find_gibbs_fidelity import get_gibbs_parameters

import tensorcircuit as tc
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")

# Number of qubits
global n
n = 4

# Get Hamiltonian
hamiltonian = tfi_hamiltonian(n)

# Path to output file
dir = 'gibbs_training_dm_seeds_xy_4.csv'
if not os.path.exists(dir):
    with open(dir, 'w') as f:
        fieldnames = ['nqubits', 'nlayers', 'nresets', 'target_energy', 'final_energy', 'infidelity', 'params',
                      'max_grad', 'norm_grad', 'iter']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Define seeds
seeds = np.random.randint(0, 500, 5)
# Define number of layers to be checked
layers = range(1, n)

# Define hyperparameter list
looping_list = []
for seed in seeds:
    for layer in layers:
        looping_list.append((layer, seed))


batch_of_jobs = [delayed(get_gibbs_parameters)(hamiltonian, n, nlayers, dir, seed=seed, maxiter=2000) for
                 nlayers, seed in looping_list]
job = Parallel(n_jobs=10)(batch_of_jobs)
