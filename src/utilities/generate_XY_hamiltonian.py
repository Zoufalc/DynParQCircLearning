# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""XY Hamiltonian Generation."""

import numpy as np
import tensorcircuit as tc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
dtype = np.complex128

xx = tc.gates._xx_matrix  # xx gate matrix to be utilized

def xy_hamiltonian(n: int,
                   dense: bool = True,
                   individual_paulis: bool = False):
    """
    :param n: Number of qubits
    :param dense: Boolean value that determines whether the Hamiltonian is returned in dense or sparse format
    :param individual_paulis: Boolean value that determines whether the individual Paulis making up the Hamiltonian are
                              returned alongside the complete Hamiltonian
    :return: XY Hamiltonian on n qubits
    """
    h = []
    w = []
    if dense:
        f = tc.quantum.PauliStringSum2Dense
    else:
        f = tc.quantum.PauliStringSum2COO_numpy

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
        w.append(-0.75)  # weight
        for j in range(n):
            if j == (i + 1) % n or j == i:
                h[i + n].append(1)
            else:
                h[i + n].append(0)

    ### YY
    for i in range(n):
        h.append([])
        w.append(-0.25)  # weight
        for j in range(n):
            if j == (i + 1) % n or j == i:
                h[i + 2*n].append(2)
            else:
                h[i + 2*n].append(0)

    hamiltonian= f(
        h, w
    )

    if individual_paulis:
        z_array = []
        z_pauli = jnp.asarray([[1, 0], [0, -1]])
        paulis = []
        for i in range(n):
            z_array.append([i])
            paulis.append(z_pauli)

        xx_array = []
        x_pauli = jnp.asarray([[0, 1], [1, 0]])
        xx_pauli = jnp.asarray(np.kron(x_pauli, x_pauli))

        for i in range(n):
            xx_array.append([i, (i + 1) % n])
            paulis.append(xx_pauli)

        yy_array = []
        y_pauli = jnp.asarray([[0, -1j], [1j, 0]])
        yy_pauli = jnp.asarray(np.kron(y_pauli, y_pauli))

        for i in range(n):
            yy_array.append([i, (i + 1) % n])
            paulis.append(yy_pauli)
        pauli_indices = z_array + xx_array + yy_array

        # Improve infidelities?

        x_array = []
        for i in range(n):
            x_array.append([i])
            paulis.append(x_pauli)
        pauli_indices += x_array

        y_array = []
        for i in range(n):
            y_array.append([i])
            paulis.append(y_pauli)
        pauli_indices += y_array

        zz_array = []
        zz_pauli = jnp.asarray(np.kron(z_pauli, z_pauli))
        for i in range(n):
            zz_array.append([i, (i + 1) % n])
            paulis.append(zz_pauli)
        pauli_indices += zz_array

        xyz_array = []
        xyz_pauli = jnp.asarray(np.kron(x_pauli, np.kron(y_pauli, z_pauli)))
        for i in range(n):
            xyz_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(xyz_pauli)

        pauli_indices += xyz_array

        xxx_array = []
        xxx_pauli = jnp.asarray(np.kron(x_pauli, np.kron(x_pauli, x_pauli)))
        for i in range(n):
            xxx_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(xxx_pauli)

        pauli_indices += xxx_array

        zzz_array = []
        zzz_pauli = jnp.asarray(np.kron(z_pauli, np.kron(z_pauli, z_pauli)))
        for i in range(n):
            zzz_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(zzz_pauli)

        pauli_indices += zzz_array

        yyy_array = []
        yyy_pauli = jnp.asarray(np.kron(y_pauli, np.kron(y_pauli, y_pauli)))
        for i in range(n):
            yyy_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(yyy_pauli)

        pauli_indices += yyy_array

        zxz_array = []
        zxz_pauli = jnp.asarray(np.kron(z_pauli, np.kron(x_pauli, z_pauli)))
        for i in range(n):
            zxz_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(zxz_pauli)

        pauli_indices += zxz_array

        yxy_array = []
        yxy_pauli = jnp.asarray(np.kron(y_pauli, np.kron(x_pauli, y_pauli)))
        for i in range(n):
            yxy_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(yxy_pauli)

        pauli_indices += yxy_array

        yzy_array = []
        yzy_pauli = jnp.asarray(np.kron(y_pauli, np.kron(z_pauli, y_pauli)))
        for i in range(n):
            yzy_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(yzy_pauli)

        pauli_indices += yzy_array

        zyz_array = []
        zyz_pauli = jnp.asarray(np.kron(z_pauli, np.kron(y_pauli, z_pauli)))
        for i in range(n):
            zyz_array.append([i, (i + 1) % n, (i + 2) % n])
            paulis.append(zyz_pauli)

        pauli_indices += zyz_array

        return hamiltonian, pauli_indices, paulis
    else:
        return hamiltonian

