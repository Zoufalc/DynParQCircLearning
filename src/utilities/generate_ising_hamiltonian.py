# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transverse field Ising Hamiltonian Generation."""

import numpy as np
import tensorcircuit as tc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
dtype = np.complex128

xx = tc.gates._xx_matrix  # xx gate matrix to be utilized

def tfi_hamiltonian(n: int,
                    n_resets: int = 0,
                    extend: bool = False, #TODO: What does this do?
                    dense: bool = True,
                    individual_paulis: bool = False):
    """

    :param n: Number of qubits.
    :param n_resets: Number of ancillary qubits used to purify the reset operation.
    :param extend:
    :param dense: Boolean value that determines whether the Hamiltonian is returned in dense or sparse format
    :param individual_paulis: Boolean value that determines whether the individual Paulis making up the Hamiltonian are
                              returned alongside the complete Hamiltonian
    :return: Transverse field Ising Hamiltonian on n qubits
    """
    h = []
    w = []

    ### Z
    for i in range(n):
        h.append([])
        w.append(-0.5)  # weight
        for j in range(n):
            if j == i:
                h[i].append(3)
            else:
                h[i].append(0)
    ### XX
    for i in range(n):
        h.append([])
        w.append(-1.0)  # weight
        for j in range(n):
            if j == (i + 1) % n or j == i:
                h[i + n].append(1)
            else:
                h[i + n].append(0)

    ### I^n
    if dense:
        f = tc.quantum.PauliStringSum2Dense
    else:
        f = tc.quantum.PauliStringSum2COO_numpy
    if extend:
        for element in h:
            for k in range(n_resets):
                element.append(0)
        hamiltonian = f(h,w)
    else:
        hamiltonian = f(h, w)

    if individual_paulis:
        z_array = []
        z_pauli = jnp.asarray([[1, 0], [0, -1]])
        paulis = []
        for i in range(n):
            z_array.append([i])
            paulis.append(z_pauli)
        x_array = []
        x_pauli = jnp.asarray([[0, 1], [1,0]])
        xx_pauli = jnp.asarray(np.kron(x_pauli, x_pauli))

        for i in range(n-1):
            x_array.append([i, i+1])
            paulis.append(xx_pauli)
        pauli_indices = z_array+x_array


        return hamiltonian, pauli_indices, paulis
    else:
        return hamiltonian
