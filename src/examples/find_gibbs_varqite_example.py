# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Gibbs state preparation based on variational quantum imaginary time evolution using the dynamic
parameterized quantum ansatz for a transverse field Ising model."""


import numpy as np
import os
import csv

np.set_printoptions(legacy='1.21')

from src.utilities.generate_ansatz import *
from src.utilities.generate_XY_hamiltonian import xy_hamiltonian

from src.find_gibbs_varqite import *

import tensorcircuit as tc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")

global n
# Number of qubits
n = 6

# Inverse temperature
beta = 0.1
h_dense = xy_hamiltonian(n)

# Path to output file
dir = 'test_varqite_jax_xy_sle.csv'

rho = K.expm(-1 * beta * h_dense)
target = rho / K.trace(rho)
target_energy = K.real(K.trace(h_dense @ target))

if os.path.exists(dir) == False:
    with open(dir, 'w', newline='') as csvfile:
        fieldnames = ['n', 'beta', 'layers', 'num_params', 'init infidelity max mixed', 'final infidelity',
                      'target energy', 'initial energy', 'reached energy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

rand_inits = np.random.randint(0, high=999, size=10)

# In this example, we want to test the performance for different layer depths
n_layers = range(1, 3)

for layers in n_layers:
    num_params = construct_dissipative_ansatz_dm(n, layers, dm_varqte=True, return_param_counter=True)
    def ansatz(params):
        return construct_dissipative_ansatz_dm(n, layers, param=params, dm_varqte=True)
    for rand_init in rand_inits:
        np.random.seed(rand_init)
        params = np.random.rand(num_params)
        for j in range(1, n+1):
            params[-j] = 3/4

        max_mixed = jnp.eye(2**n)/2**n
        inf_max_mixed = infidelity(params, ansatz, max_mixed)
        # Run state preparation
        gibbs_parameters, final_energy, initial_energy = get_gibbs(h_dense, ansatz, tc.array_to_tensor(params),
                                                        inv_temperature=beta)
        # Check preparation performance
        infid_final = infidelity(gibbs_parameters, ansatz, target) #[-1][0]
        # Store outcome
        with open(dir, 'a', newline='') as csvfile:
            fieldnames = ['n', 'beta', 'layers', 'num_params', 'init infidelity max mixed', 'final infidelity',
                         'target energy', 'initial energy', 'reached energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'n': n,
                             'beta': beta,
                             'layers': layers,
                             'num_params': len(params),
                             'init infidelity max mixed': inf_max_mixed,
                             'final infidelity': infid_final,
                             'target energy': target_energy,
                             'initial energy': initial_energy,
                             'reached energy': final_energy})