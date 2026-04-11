# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for Toric Code Hamiltonian using the dynamic parameterized quantum circuit ansatz."""

from src.utilities.generate_toric_code_hamiltonian import *
from src.utilities.generate_ansatz import *
from src.utilities.ansatz_classes import ToricCodeAnsatz, OneDBrickwork
from jax import numpy as jnp

# Define system parameters
Lx = 2
Ly = 2
num_qubits=8
nlayers = num_qubits
howoften_toreset = nlayers-1
howoften_tosave = 10
# tc_ = ToricCode(Lx,Ly)
h = 0.1
trials = 10
maxiter = 501
learning_rate = 1e-2
unitary = False # If true, gives a purely unitary ansatz that does not have any ancillas on the plaquettes. Else the usual one we have in the paper.

if __name__ == "__main__":
    ansatz = ToricCodeAnsatz(
        Lx=Lx,
        Ly=Ly,
        nlayers=nlayers,
        howoften_toreset=howoften_toreset,
        h=h,
        trials=trials,
        maxiter=maxiter,
        howoften_tosave=howoften_tosave,
        unitary=True,
        sparse=False,
        perform_noisy_simulations=True,
        noise_rate=5e-2,
        number_of_shots=2000
    )
    final_energies, final_parameters, all_energy_values, all_purity_values = ansatz.optimize(save_results=True)
    print('final energies ', final_energies)

    # initial_values = ansatz.get_initial_costs()
    initial_values = all_energy_values[:,0]
    print('Initial energies mean: ', jnp.mean(initial_values), '\nInitial energies variance: ', jnp.var(initial_values))
    print("Final energies mean: ", jnp.mean(final_energies), "\nFinal energies variance: ", jnp.var(final_energies))


    ansatz = ToricCodeAnsatz(
        Lx=Lx,
        Ly=Ly,
        nlayers=nlayers,
        howoften_toreset=howoften_toreset,
        h=h,
        trials=trials,
        maxiter=maxiter,
        howoften_tosave=howoften_tosave,
        use_prob_resets=False,
        sparse=False,
        perform_noisy_simulations=True,
        noise_rate=5e-2,
        number_of_shots=2000,
        max_trials_in_batch=1
        )
    final_energies, final_parameters, all_energy_values, all_purity_values = ansatz.optimize(save_results=True)
    print('final energies ', final_energies)

    # initial_values = ansatz.get_initial_costs()
    initial_values = all_energy_values[:,0]
    print('Initial energies mean: ', jnp.mean(initial_values), '\nInitial energies variance: ', jnp.var(initial_values))
    print("Final energies mean: ", jnp.mean(final_energies), "\nFinal energies variance: ", jnp.var(final_energies))