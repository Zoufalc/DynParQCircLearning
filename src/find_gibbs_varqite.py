# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational Gibbs state preparation based on variational quantum imaginary time evolution."""

import numpy as np

from typing import Iterable

np.set_printoptions(legacy='1.21')

from src.utilities.generate_ansatz import *
from src.utilities.gibbs_varqite import VarQTE
from src.utilities.tc_grads import *

import tensorcircuit as tc

import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")


def infidelity(params: Iterable,
               ansatz: tc.Circuit,
               target_dens: Iterable) -> float:
    """
    Compute infidelity between prepared and target state.
    :param params: Ansatz parameter values.
    :param ansatz: Ansatz circuit corresponding to density matrix.
    :param target_dens: Target state in the form of a density matrix.
    :return: infidelity
    """
    state = ansatz(params)
    prep_dens = state.densitymatrix()
    fid = K.trace(K.sqrtmh(prep_dens @ target_dens, psd=True))**2
    fid = K.real(fid)
    infid = 1 - fid
    return K.real(infid)

def get_gibbs(hamiltonian: SparsePauliOp,
              ansatz: tc.Circuit,
              parameters: Iterable,
              inv_temperature: float):
    """
    Get a Gibbs state
    :param hamiltonian: Hamiltonian to find the Gibbs state for.
    :param ansatz: Parameterized ansatz for Gibbs state preparation.
    :param parameters: Initial values for ansatz parameters.
    :param inv_temperature: Inverse temperature for Gibbs state preparation.
    :return: Gibbs state, final energy, initial energy
    """
    circ = ansatz(parameters)
    sys_state = circ.densitymatrix()
    energy = (hamiltonian @ sys_state).trace()
    # Get initial energy
    initial_energy = K.real(energy)
    # Define variational quantum time evolution
    varqite = VarQTE(ansatz=ansatz, initial_parameters=[parameters],
                     gradient=grad_dm_tc(ansatz, (-1)*hamiltonian, backend=K),
                     qgt=qgt_dm_tc(ansatz, backend=K), dm=True, backend=K)
    #Evolve the state using variational quantum imaginary time evolution
    gibbs_parameter_values = varqite.evolve(hamiltonian, final_time=inv_temperature/2, stepsize=0.01, res_=False)
    state = ansatz(gibbs_parameter_values[-1][0]).densitymatrix()
    energy = (hamiltonian @ state).trace()
    final_energy = K.real(energy)
    return gibbs_parameter_values[-1][0], final_energy, initial_energy
