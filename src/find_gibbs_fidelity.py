# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Gibbs state preparation based on training the infidelity."""

import numpy as np
import csv
from typing import Iterable

np.set_printoptions(legacy='1.21')

import tensorcircuit as tc

import jax
jax.config.update("jax_enable_x64", True)
import optax
K = tc.set_backend("jax")
from qiskit.quantum_info import SparsePauliOp

from src.utilities.generate_ansatz import *


def fidelity(prep_state: Iterable,
             target_dens: Iterable,
             n_aux: int) -> float:
    """
    Evaluate fidelity
    :param prep_state: Purified prepared state.
    :param target_dens: Target density matrix.
    :param n_aux: Number of auxiliary qubits.
    :return: fidelity
    """
    dens = np.kron(target_dens, np.eye(2**n_aux))
    prep_dens = np.outer(prep_state, np.transpose(np.conj(prep_state)))
    fidelity_estimate = np.trace(np.sqrt(np.dot(np.sqrt(dens), np.dot(prep_dens, np.sqrt(dens)))))
    return fidelity_estimate

def get_gibbs_energy(hamiltonian: SparsePauliOp,
                     circuit: tc.Circuit,
                     params: Iterable) -> float:
    """
    Get energy for Gibbs state
    :param hamiltonian: Hamiltonian underlying Gibbs state preparation.
    :param circuit: Ansatz circuit.
    :param params: Values for ansatz parameters corresponding to Gibbs state.
    :return: energy
    """
    circ = circuit(params)
    sys_state = circ.densitymatrix()
    energy = (hamiltonian @ sys_state).trace()
    energy = K.real(energy)
    return energy

def get_gibbs_parameters(hamiltonian: Iterable,
                         n: int,
                         nlayers: int,
                         dir: str,
                         seed = 230,
                         maxiter=10000) -> Iterable:
    """
    Train ansatz parameters to approximate Gibbs state by minimizing the infidelity
    :param hamiltonian: Hamiltonian for Gibbs state preparation.
    :param n: Number of qubits.
    :param nlayers: Number of layers to be used in the ansatz.
    :param dir: Directory to store results at.
    :param seed: Seed for random iterations.
    :param maxiter: Maximum number of iterations.
    :return: Parameter values for the underlying ansatz which corresponds to the final Gibbs state approximation.
    """

    def ansatz(params):
        return construct_dissipative_ansatz_dm(n, nlayers, param=params)

    def infidelity(params, target_dens, print_traces=False):
        state = ansatz(params)
        prep_dens = state.densitymatrix()
        fid = K.trace(K.sqrtmh(prep_dens @ target_dens, psd=True)) ** 2
        fid = K.real(fid)
        infid = 1 - fid
        if print_traces:
            # print('params ', params)
            print('Trace density matrix ', K.trace(prep_dens))
            print('Target density matrix ', K.trace(target_dens))
        return K.real(infid)  # tc_inf

    f_grad = tc.backend.jit(
        tc.backend.value_and_grad(infidelity), static_argnums=(2)
    )
    beta = 2
    # h_dense = xy_hamiltonian(n)
    rho = K.expm(-beta * hamiltonian)
    target = rho / K.trace(rho)
    target_energy = K.real(K.trace(hamiltonian @ target))
    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(key,
        shape=[int(nlayers * (12 * (n - 2) + 20 * n))],#minval=0,maxval=0.01
    )

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    params = jnp.array(params)
    opt_state = optimizer.init(params)

    for i in range(maxiter):
        if np.any(np.isnan(params)):
            params = np.nan_to_num(params)
        (value, gradient) = f_grad(params, target_dens=target)
        updates, opt_state = optimizer.update(gradient, opt_state)
        if value < 0:
            break
        if np.any(np.isnan(params)):
            break
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            if np.any(np.isnan(params)):
                params = np.nan_to_num(params)
            if np.any(np.isnan(gradient)):
                gradient = np.nan_to_num(gradient)
            gibbs_energy = get_gibbs_energy(hamiltonian, ansatz, params)
            row = {'nqubits': n, 'nlayers': nlayers, 'nresets': n - 1, 'target_energy': target_energy,
                   'final_energy': gibbs_energy, 'infidelity': value, 'params': params,
                   'max_grad': np.max(gradient), 'norm_grad': np.linalg.norm(gradient), 'iter': i}
            with open(dir, 'a') as f:
                fieldnames = ['nqubits', 'nlayers', 'nresets', 'target_energy', 'final_energy', 'infidelity',
                              'params', 'max_grad', 'norm_grad', 'iter']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
    return params

# def get_gibbs_parameters(hamiltonian: Iterable,
#                          n: int,
#                          nlayers: int,
#                          dir: str,
#                          seed = 230,
#                          maxiter=10000) -> Iterable:
#     """
#     Train ansatz parameters to approximate Gibbs state by minimizing the infidelity
#     :param hamiltonian: Hamiltonian for Gibbs state preparation.
#     :param n: Number of qubits.
#     :param nlayers: Number of layers to be used in the ansatz.
#     :param dir: Directory to store results at.
#     :param seed: Seed for random iterations.
#     :param maxiter: Maximum number of iterations.
#     :return: Parameter values for the underlying ansatz which corresponds to the final Gibbs state approximation.
#     """
#
#     if not os.path.exists(dir):
#         with open(dir, 'w') as f:
#             fieldnames = ['nqubits', 'nlayers', 'nresets', 'target_energy', 'final_energy', 'infidelity', 'params',
#                           'max_grad', 'norm_grad', 'iter']
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#
#     def ansatz(params):
#         return construct_dissipative_ansatz_dm(n, nlayers, param=params)
#
#     def infidelity(params, target_dens):
#         state = ansatz(params)
#         prep_dens = state.densitymatrix()
#         fid = K.trace(K.sqrtmhpos(prep_dens @ target_dens)) ** 2
#         fid = K.real(fid)
#         infid = 1 - fid
#         return K.real(infid)
#
#     f_grad = tc.backend.jit(
#         tc.backend.value_and_grad(infidelity), static_argnums=(1)
#     )
#     beta = 2
#     rho = K.expm(-beta * hamiltonian)
#     target = rho / K.trace(rho)
#     target_energy = K.real(K.trace(hamiltonian @ target))
#     print('target energy ', target_energy)
#     key = jax.random.PRNGKey(seed)
#     params = jax.random.uniform(key,
#         shape=[int(nlayers * (12 * (n - 2) + 20 * n))], #minval=0,maxval=0.01
#     )
#
#     learning_rate = 1e-2
#     optimizer = optax.adam(learning_rate)
#     params = jnp.array(params)
#     opt_state = optimizer.init(params)
#
#     for i in range(maxiter):
#         if np.any(np.isnan(params)):
#             params = np.nan_to_num(params)
#         (value, gradient) = f_grad(params, target_dens=target)
#         updates, opt_state = optimizer.update(gradient, opt_state)
#         if value < 0:
#             break
#         if np.any(np.isnan(params)):
#             break
#         params = optax.apply_updates(params, updates)
#         if i % 10 == 0:
#             if np.any(np.isnan(params)):
#                 params = np.nan_to_num(params)
#             if np.any(np.isnan(gradient)):
#                 gradient = np.nan_to_num(gradient)
#             gibbs_energy = get_gibbs_energy(hamiltonian, ansatz, params)
#             row = {'nqubits': n, 'nlayers': nlayers, 'nresets': n - 1, 'target_energy': target_energy,
#                    'final_energy': gibbs_energy, 'infidelity': value, 'params': params,
#                    'max_grad': np.max(gradient), 'norm_grad': np.linalg.norm(gradient), 'iter': i}
#             with open(dir, 'a') as f:
#                 fieldnames = ['nqubits', 'nlayers', 'nresets', 'target_energy', 'final_energy', 'infidelity',
#                               'params', 'max_grad', 'norm_grad', 'iter']
#                 writer = csv.DictWriter(f, fieldnames=fieldnames)
#                 writer.writerow(row)
#     return params
