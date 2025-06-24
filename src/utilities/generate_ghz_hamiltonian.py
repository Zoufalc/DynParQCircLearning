# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""GHZ Hamiltonian Generation."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli

def ghz_hamiltonian(n: int) -> SparsePauliOp:
       """
       This function prepares a Hamiltonian for which an n-qubit GHZ state corresponds to the ground state
       :param n: number qubits
       :return: parent Hamiltonian for a GHZ state on n qubits
       """
       zero = SparsePauliOp(['I', 'Z'], np.array([1/2, 1/2]))
       one = SparsePauliOp(['I', 'Z'], np.array([1/2, -1/2]))
       plus = SparsePauliOp(['I', 'X'], np.array([1/2, 1/2]))

       zero3 = zero.tensor(zero.tensor(zero))
       one3 = one.tensor(one.tensor(one))
       zero_plus_one = one.tensor(plus.tensor(zero))
       one_plus_zero = zero.tensor(plus.tensor(one))

       h_op = Pauli('III') - zero3 - one3 - one_plus_zero - zero_plus_one
       h_op = h_op.simplify()
       H = SparsePauliOp('I'*(n-3)).tensor(h_op).simplify()
       for k in range(1, n-2):
              pre_fill = SparsePauliOp('I'*k)
              post_fill = SparsePauliOp('I'*(n-3-k))
              H += post_fill.tensor(h_op.tensor(pre_fill))
       return H.simplify()


